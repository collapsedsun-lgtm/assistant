import argparse
import asyncio
import json
import time
from typing import List

import aiohttp

import actions
from plugin_loader import load_plugins
import memory_summarizer
import web_sanitizer
from kv_store import get_default_kv, response_cache_key
import session as session_module
from llm_client import call_llm, check_model_endpoint
from utils import load_system_prompt, load_settings, _sanitize_assistant_output
from assistant_core import build_messages, try_parse_tool_call
from assistant import ROLLING_WINDOW
from tts_piper import PiperTTS
from tts_playback import PlaybackManager
import tempfile
import time
import os
import shutil
import subprocess


async def ainput(prompt: str = "") -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, input, prompt)


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use a mock LLM response for testing")
    parser.add_argument("--run-plugins", action="store_true", help="Allow executing discovered plugins for validated actions")
    parser.add_argument("--debug", action="store_true", help="Enable debug output for LLM requests")
    parser.add_argument("--rolling-window", type=int, default=ROLLING_WINDOW, help="Number of recent exchanges to include in context (user+assistant pairs)")
    parser.add_argument("--summarize-memory", action="store_true", help="Enable memory summarization of recent exchanges before sending to the LLM")
    parser.add_argument("--show-settings", action="store_true", help="Print effective LLM and cache settings and exit")
    args = parser.parse_args()
    # Allow enabling plugin execution by default via settings.json
    try:
        settings = load_settings()
        if not args.run_plugins and isinstance(settings, dict) and settings.get("auto_run_plugins"):
            args.run_plugins = True
    except Exception:
        pass

    async def _print_settings_and_exit():
        # Load KV to report backend status
        try:
            kv = await get_default_kv()
            backend = "redis" if getattr(kv, "_use_redis", False) else "in-memory"
        except Exception:
            backend = "unavailable"

        # Basic validation
        warnings = []
        from llm_config import LLM_STREAM, PARTIAL_TTL, FINAL_TTL, PARTIAL_SAVE_THRESHOLD, MODEL, OLLAMA_URL, LLM_OPTIONS

        if PARTIAL_TTL <= 0:
            warnings.append("PARTIAL_TTL should be > 0")
        if FINAL_TTL <= 0:
            warnings.append("FINAL_TTL should be > 0")
        if PARTIAL_SAVE_THRESHOLD <= 0:
            warnings.append("PARTIAL_SAVE_THRESHOLD should be > 0")

        print("Effective settings:")
        print(f"- model: {MODEL}")
        print(f"- ollama_url: {OLLAMA_URL}")
        print(f"- llm_options: {json.dumps(LLM_OPTIONS)}")
        print(f"- streaming enabled (env/settings): {LLM_STREAM}")
        print(f"- partial_ttl: {PARTIAL_TTL}s")
        print(f"- final_ttl: {FINAL_TTL}s")
        print(f"- partial_save_threshold: {PARTIAL_SAVE_THRESHOLD} chars")
        print(f"- kv backend: {backend}")
        if warnings:
            print("Warnings:")
            for w in warnings:
                print(f"- {w}")
        return

    if args.show_settings:
        await _print_settings_and_exit()
        return

    history: List[dict] = []
    actions_list = actions.load_actions()
    settings = load_settings()
    handlers, pre_send_hooks, on_start_hooks = load_plugins(settings)

    # Initialize TTS (Piper) and optional playback manager if configured
    tts_settings = settings.get("tts", {}) if isinstance(settings, dict) else {}
    _tts_enabled = bool(tts_settings.get("enabled", False))
    _tts_provider = tts_settings.get("provider", "piper")
    piper_tts = None
    playback_manager = None
    if _tts_enabled and _tts_provider == "piper":
        try:
            piper_mode = tts_settings.get("mode", "http")
            piper_server = tts_settings.get("server_url")
            piper_bin = tts_settings.get("binary_cmd")
            piper_headers = tts_settings.get("headers", {})
            piper_tts = PiperTTS(mode=piper_mode, server_url=piper_server, binary_cmd=piper_bin, headers=piper_headers)
            print("TTS enabled: Piper configured")
        except Exception as e:
            print("Failed to initialize Piper TTS:", e)

        # Playback manager initialization
        try:
            pb = tts_settings.get("playback", {})
            if pb.get("enabled", True):
                preferred = pb.get("preferred_player")
                volume = pb.get("volume", 1.0)
                playback_manager = PlaybackManager(preferred_player=preferred, default_volume=volume, queue_enabled=pb.get("queue", True))
                print("TTS playback enabled")
        except Exception as e:
            print("Failed to initialize playback manager:", e)

    async def _maybe_speak(text: str):
        if not piper_tts:
            return
        out_dir = tts_settings.get("output_dir", "/tmp")
        try:
            fn = os.path.join(out_dir, f"assistant_tts_{int(time.time()*1000)}.wav")
            loop = asyncio.get_running_loop()

            # For HTTP-mode Piper we can avoid disk I/O and stream-play raw WAV bytes
            if getattr(piper_tts, "mode", None) == "http":
                try:
                    audio_bytes = await loop.run_in_executor(None, piper_tts.synthesize, text, None)
                    # If a playback manager is configured, enqueue bytes (non-blocking)
                    if playback_manager:
                        playback_manager.enqueue_bytes(audio_bytes)
                        print("[TTS enqueued for playback]")
                        return

                    # No manager: try direct player piping (best-effort)
                    played = False
                    if shutil.which("aplay"):
                        p = subprocess.Popen(["aplay", "-t", "wav", "-"], stdin=subprocess.PIPE)
                        p.stdin.write(audio_bytes)
                        p.stdin.close()
                        played = True
                    elif shutil.which("ffplay"):
                        p = subprocess.Popen(["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", "-i", "-"], stdin=subprocess.PIPE)
                        p.stdin.write(audio_bytes)
                        p.stdin.close()
                        played = True

                    if played:
                        print("[TTS played (stream)]")
                        return

                    # fallback: write to file and play via xdg-open
                    with open(fn, "wb") as fh:
                        fh.write(audio_bytes)
                    print(f"[TTS saved to {fn}]")
                    if shutil.which("xdg-open"):
                        subprocess.Popen(["xdg-open", fn])
                    return
                except Exception:
                    # proceed to file-based fallback
                    pass

            # Binary-mode: attempt to capture stdout bytes from the CLI binary
            if getattr(piper_tts, "mode", None) == "binary":
                try:
                    audio_bytes = await loop.run_in_executor(None, piper_tts.synthesize, text, None)
                    if isinstance(audio_bytes, (bytes, bytearray)):
                        if playback_manager:
                            playback_manager.enqueue_bytes(bytes(audio_bytes))
                            print("[TTS enqueued for playback]")
                            return
                        # No manager: attempt direct piping
                        played = False
                        if shutil.which("aplay"):
                            p = subprocess.Popen(["aplay", "-t", "wav", "-"], stdin=subprocess.PIPE)
                            p.stdin.write(audio_bytes)
                            p.stdin.close()
                            played = True
                        elif shutil.which("ffplay"):
                            p = subprocess.Popen(["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", "-i", "-"], stdin=subprocess.PIPE)
                            p.stdin.write(audio_bytes)
                            p.stdin.close()
                            played = True

                        if played:
                            print("[TTS played (stream)]")
                            return

                        # fallback to file
                        with open(fn, "wb") as fh:
                            fh.write(audio_bytes)
                        print(f"[TTS saved to {fn}]")
                        if shutil.which("xdg-open"):
                            subprocess.Popen(["xdg-open", fn])
                        return
                except Exception:
                    # fall back to file-based invocation
                    pass

            # Default: write to file (fallback) and use playback manager if available
            await loop.run_in_executor(None, piper_tts.synthesize, text, fn)
            print(f"[TTS saved to {fn}]")
            if playback_manager:
                playback_manager.enqueue_file(fn)
            else:
                if shutil.which("aplay"):
                    subprocess.Popen(["aplay", fn])
                elif shutil.which("ffplay"):
                    subprocess.Popen(["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", fn])
                elif shutil.which("xdg-open"):
                    subprocess.Popen(["xdg-open", fn])
        except Exception as e:
            print("[TTS error]:", e)

    # Connect to KV and load recent history (if any). Uses a simple default
    # session id; this can be extended later to support per-user sessions.
    session_id = "default"
    kv, history = await session_module.load_history(session_id)
    if handlers:
        print("\nDiscovered plugin handlers (action -> handler):")
        for name in sorted(handlers.keys()):
            print(f"- {name}")
    else:
        print("\nNo plugins discovered in plugins/ directory.")

    if pre_send_hooks:
        print(f"\nDiscovered {len(pre_send_hooks)} pre_send hooks available from plugins.")

    if on_start_hooks:
        print(f"\nRunning {len(on_start_hooks)} startup hooks...")
        for hook in on_start_hooks:
            try:
                maybe = None
                try:
                    import inspect
                    sig = inspect.signature(hook)
                    kwargs = {}
                    params = sig.parameters
                    if "debug" in params:
                        kwargs["debug"] = args.debug
                    if "settings" in params or "config" in params:
                        kwargs["settings"] = settings
                    maybe = await hook(**kwargs)
                except Exception:
                    try:
                        maybe = await hook()
                    except Exception:
                        maybe = None

                if args.debug:
                    print(f"[debug] startup hook {getattr(hook, '__name__', repr(hook))} returned: {repr(maybe)[:300]}")
            except Exception as e:
                if args.debug:
                    print(f"[debug] startup hook failed: {e}")

    # If not running in mock mode, perform a lightweight health check against the model
    if not args.mock:
        ok, info = await check_model_endpoint(debug=args.debug)
        if ok:
            print("\nOllama connectivity check OK:", info)
        else:
            print("\nOllama connectivity check failed:", info)
            ans = await ainput("Continue without LLM (you can use --mock)? (y/N): ")
            if ans.strip().lower() not in ("y", "yes"):
                print("Exiting due to failed Ollama connectivity check.")
                return
    else:
        print("\nSkipping LLM health check because --mock was provided.")

    # Session mode: one-time bootstrap turn (system + user), then rely on
    # server session memory. For `/api/chat`, Ollama does not provide stable
    # cross-request server-side conversation state, so we skip session mode.
    from assistant import OLLAMA_SESSION_ENABLED, OLLAMA_SESSION_BOOTSTRAPPED
    global OLLAMA_SESSION_ENABLED, OLLAMA_SESSION_BOOTSTRAPPED
    if settings.get("ollama_use_session", False):
        from llm_config import OLLAMA_URL as _OLLAMA_URL
        if _OLLAMA_URL.rstrip("/").endswith("/api/chat"):
            OLLAMA_SESSION_ENABLED = False
            OLLAMA_SESSION_BOOTSTRAPPED = False
            print("Session mode requested, but /api/chat is stateless across requests in this client flow.")
            print("Falling back to per-request prompts/history for reliable behavior.")
        else:
            OLLAMA_SESSION_ENABLED = True
            if args.debug:
                print("[debug] session mode enabled; running one-time bootstrap turn")

            sys_prompt = load_system_prompt()
            adesc = actions.actions_description(actions_list)
            bootstrap_messages = [
                {"role": "system", "content": sys_prompt + "\n\nAvailable actions:\n" + adesc},
                {
                    "role": "user",
                    "content": "From now on, the user will interact directly. Reply with READY.",
                },
            ]
            bootstrap_payload = {"model": MODEL, "messages": bootstrap_messages}
            from llm_config import LLM_OPTIONS, LLM_TIMEOUT
            bootstrap_payload.update(LLM_OPTIONS)
            if "keep_alive" not in bootstrap_payload:
                bootstrap_payload["keep_alive"] = settings.get("ollama_keep_alive", "30m")

            timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT)
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(OLLAMA_URL, json=bootstrap_payload) as bresp:
                        body = await bresp.text()
                        if bresp.status != 200:
                            print(f"Session bootstrap failed: status={bresp.status} body={body[:400]}")
                            print("Exiting because session bootstrap is required when ollama_use_session=true.")
                            return
                OLLAMA_SESSION_BOOTSTRAPPED = True
                if args.debug:
                    print("[debug] session bootstrap succeeded; subsequent requests will use session-mode payloads")
            except Exception as e:
                print(f"Session bootstrap failed with exception: {e}")
                print("Exiting because session bootstrap is required when ollama_use_session=true.")
                return

    while True:
        try:
            user_input = await ainput("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye")
            break
        # Persist the user's message to KV immediately (write-through) for
        # crash recovery, but do NOT append it to the in-memory `history`
        # before calling the LLM — `build_messages` already adds the
        # current `user_input` as the final message, and appending it to
        # `history` causes duplication in the request payload.
        try:
            await session_module.persist_message(session_id, kv, "user", user_input)
        except Exception:
            pass

        try:
            llm_output, prefetch_texts = await call_llm(
                user_input,
                history,
                mock=args.mock,
                debug=args.debug,
                rolling_window=args.rolling_window,
                summarize_memory=args.summarize_memory,
                pre_send_hooks=pre_send_hooks,
            )
            # If a streaming generator was returned, consume it and assemble final text
            if hasattr(llm_output, "__aiter__"):
                chunks = []
                cumulative = ""
                mode = None  # 'maybe_json' or 'text'
                detected_action = False
                stream_printed_progressively = False
                try:
                    async for part in llm_output:
                        # Determine mode from first non-whitespace char of first part
                        if mode is None:
                            first_non_ws = None
                            for ch in part:
                                if not ch.isspace():
                                    first_non_ws = ch
                                    break
                            if first_non_ws in ("{", "["):
                                mode = "maybe_json"
                            else:
                                mode = "text"
                                # Do not print the assistant label here; we'll print
                                # the sanitized final reply below to avoid duplicate
                                # leading labels when the model emits 'Assistant: ...'.
                                stream_printed_progressively = True

                        cumulative += part

                        if mode == "text":
                            # Buffer text parts instead of printing progressively
                            chunks.append(part)
                        else:
                            # buffer JSON-like responses and attempt to parse as a tool call
                            try:
                                parsed_check = try_parse_tool_call(cumulative, actions_list)
                                if parsed_check:
                                    # Detected a tool-call JSON response; stop streaming and
                                    # treat this as an action.
                                    detected_action = True
                                    # ensure llm_output becomes the JSON string for downstream parsing
                                    llm_output = cumulative
                                    break
                            except Exception:
                                # not parseable yet; continue buffering
                                pass
                except Exception as e:
                    if mode == "text":
                        print("\n[streaming error]:", e)
                    else:
                        print("[streaming error]:", e)

                # If we didn't detect an action and were in JSON-mode, treat cumulative as text
                if not detected_action:
                    if mode == "maybe_json":
                        # treat the buffered content as text
                        final_content = cumulative
                    else:
                        final_content = "".join(chunks)

                # Cache the final streaming response when possible
                try:
                    sys_prompt = load_system_prompt()
                    msgs_for_cache = build_messages(sys_prompt, actions_desc, history, user_input, args.rolling_window, args.summarize_memory, prefetch_texts=prefetch_texts)
                    messages_json = json.dumps(msgs_for_cache, sort_keys=True)
                    cache_key = response_cache_key(MODEL, LLM_OPTIONS, messages_json)
                    if kv is not None and cache_key and not detected_action:
                        try:
                            await kv.set(cache_key, final_content, ex=FINAL_TTL)
                        except Exception:
                            pass
                except Exception:
                    pass

                if not detected_action:
                    llm_output = final_content
        except Exception as e:
            print("Assistant (error):", e)
            continue

        parsed = try_parse_tool_call(llm_output, actions_list)
        # If we have sanitized pre-fetched facts and the user's request looks
        # informational (e.g., weather, definition), prefer returning the
        # sanitized facts as a plain-text assistant reply rather than relying
        # on the model to self-report data access.
        informational_keywords = ("weather", "forecast", "rain", "define", "what is", "who is", "tell me", "info", "information", "time", "date")
        if prefetch_texts and (not parsed or getattr(parsed, "tool", "") == "null"):
            low = (user_input or "").lower()
            if any(k in low for k in informational_keywords):
                is_weather_query = any(k in low for k in ("weather", "forecast", "rain", "tomorrow", "week", "sunny"))
                is_time_query = any(k in low for k in ("time", "date", "day", "hour"))

                weather_snippets = [p for p in prefetch_texts if isinstance(p, str) and p.lower().startswith("weather (sanitized):")]
                time_snippets = [p for p in prefetch_texts if isinstance(p, str) and p.lower().startswith("time (sanitized):")]

                if is_weather_query and weather_snippets:
                    assistant_text = web_sanitizer.summarize_open_meteo_weather(weather_snippets[0], user_input)
                elif is_time_query and time_snippets:
                    assistant_text = web_sanitizer.summarize_time_context(time_snippets[0])
                else:
                    # Generic fallback: avoid dumping raw context blocks
                    friendly = []
                    for p in prefetch_texts:
                        if not isinstance(p, str):
                            continue
                        if p.lower().startswith("weather (sanitized):"):
                            friendly.append(web_sanitizer.summarize_open_meteo_weather(p, user_input))
                        elif p.lower().startswith("time (sanitized):"):
                            # include time only for explicit time/date questions
                            if is_time_query:
                                friendly.append(web_sanitizer.summarize_time_context(p))
                        else:
                            friendly.append(p)
                    assistant_text = "\n".join([x for x in friendly if x]).strip()
                    if not assistant_text:
                        assistant_text = "I do not have enough information to answer that yet."
                print("Assistant:", assistant_text)
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": assistant_text})
                try:
                    await session_module.persist_message(session_id, kv, "assistant", assistant_text)
                except Exception:
                    pass
                # Speak the assisted text when TTS is enabled
                try:
                    await _maybe_speak(assistant_text)
                except Exception:
                    pass
                continue
        if parsed:
            # Print the action JSON (the agent's intended action)
            # Use `model_dump_json` when available (pydantic v2), fall back to `.json()` for compatibility.
            if hasattr(parsed, "model_dump_json"):
                parsed_json = parsed.model_dump_json()
            else:
                parsed_json = parsed.json()
            print("Assistant (action):", parsed_json)

            # Optionally execute a matching plugin handler if allowed and available
            if args.run_plugins:
                handler = handlers.get(parsed.tool)
                if handler:
                    try:
                        result = await handler(parsed.args)
                        print("Plugin result:", result)

                        # Ask the LLM to convert the plugin result into a
                        # short, friendly assistant reply suitable for TTS.
                        try:
                            # Ensure result is serializable
                            try:
                                result_json = json.dumps(result)
                            except Exception:
                                result_json = str(result)

                            format_prompt = (
                                "Convert the following plugin result JSON into a short, "
                                "friendly assistant reply suitable for spoken output (1-2 sentences). "
                                "If the plugin result indicates an error, explain it briefly and clearly.\n\n"
                                f"Plugin result: {result_json}\n"
                                f"Original user request: {user_input}\n"
                            )

                            formatted, _ = await call_llm(
                                format_prompt,
                                history,
                                mock=args.mock,
                                debug=args.debug,
                                rolling_window=args.rolling_window,
                                summarize_memory=False,
                                pre_send_hooks=None,
                            )

                            # If streaming, consume
                            if hasattr(formatted, "__aiter__"):
                                parts = []
                                try:
                                    async for p in formatted:
                                        parts.append(p)
                                except Exception:
                                    pass
                                formatted_text = "".join(parts)
                            else:
                                formatted_text = formatted

                            # Sanitize, print and persist the formatted assistant reply
                            cleaned_formatted = _sanitize_assistant_output(formatted_text)
                            print("Assistant:", cleaned_formatted)
                            history.append({"role": "assistant", "content": cleaned_formatted})
                            try:
                                await session_module.persist_message(session_id, kv, "assistant", cleaned_formatted)
                            except Exception:
                                pass
                            try:
                                await _maybe_speak(cleaned_formatted)
                            except Exception:
                                pass
                        except Exception as e:
                            # Fall back to printing raw plugin result
                            print("Plugin formatting error:", e)
                    except Exception as e:
                        print("Plugin execution error:", e)
                else:
                    print(f"No plugin registered for action '{parsed.tool}'")

            # Save the user message then assistant reply in chronological order
            history.append({"role": "assistant", "content": parsed_json})
            try:
                await session_module.persist_message(session_id, kv, "assistant", parsed_json)
            except Exception:
                pass
            continue

        # Not a tool call: normal assistant reply
        cleaned = _sanitize_assistant_output(llm_output)
        print("Assistant:", cleaned)
        # Save the user message then assistant reply in chronological order
        history.append({"role": "assistant", "content": cleaned})
        try:
            await session_module.persist_message(session_id, kv, "assistant", cleaned)
        except Exception:
            pass
        try:
            await _maybe_speak(cleaned)
        except Exception:
            pass
