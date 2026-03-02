import argparse
import asyncio
import json
import os
from typing import List

import aiohttp
import inspect
import time

from llm_config import MODEL, OLLAMA_URL, LLM_OPTIONS
from llm_config import LLM_STREAM, PARTIAL_TTL, FINAL_TTL, PARTIAL_SAVE_THRESHOLD, LLM_TIMEOUT
import actions
from plugin_loader import load_plugins
import memory_summarizer
import web_sanitizer
from kv_store import get_default_kv, response_cache_key
import session as session_module
from llm_client import call_llm, check_model_endpoint
from llm_config import LLM_STREAM

# Session mode flag: when True we send a short hint per request and rely on
# Ollama to retain the full system prompt in its session memory after a
# one-time bootstrap. When enabled via settings we assume session support.
OLLAMA_SESSION_ENABLED = False
OLLAMA_SESSION_BOOTSTRAPPED = False


def _extract_texts(obj):
    """Extract assistant text pieces from streaming JSON payloads.

    Prefer known content-bearing keys and avoid model metadata fields such as
    `model`, `created_at`, `role`, and stop reasons.
    """
    out = []
    if obj is None:
        return out
    if isinstance(obj, str):
        out.append(obj)
        return out
    if isinstance(obj, dict):
        message = obj.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                out.append(content)

        response = obj.get("response")
        if isinstance(response, str) and response:
            out.append(response)

        delta = obj.get("delta")
        if isinstance(delta, dict):
            dcontent = delta.get("content")
            if isinstance(dcontent, str) and dcontent:
                out.append(dcontent)

        text = obj.get("text")
        if isinstance(text, str) and text:
            out.append(text)

        content = obj.get("content")
        if isinstance(content, str) and content:
            out.append(content)

        if out:
            return out

        for v in obj.values():
            out.extend(_extract_texts(v))
        return out
    if isinstance(obj, list):
        for it in obj:
            out.extend(_extract_texts(it))
        return out
    # other primitive (int/float/bool) -> ignore
    return out


def _sanitize_assistant_output(text: str) -> str:
    """Remove leading speaker labels that models sometimes emit (e.g. "Assistant:").

    This prevents duplicate printed labels like "Assistant: Assistant: ...".
    """
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    import re

    # Remove common leading speaker prefixes (case-insensitive), e.g.
    # "Assistant: ", "assistant -", "Assistant —".
    cleaned = re.sub(r"^\s*(assistant)\s*[:\-–—]\s*", "", text, flags=re.I)
    # Also remove a bare leading 'assistant' followed by whitespace/newline
    cleaned = re.sub(r"^\s*(assistant)\s+", "", cleaned, flags=re.I)
    return cleaned

# Number of most-recent exchanges (user + assistant) to include in context.
# This is a hardcoded constant for now; make configurable later if needed.
ROLLING_WINDOW = 5


def load_system_prompt() -> str:
    path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_settings() -> dict:
    path = os.path.join(os.path.dirname(__file__), "settings.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_messages(
    system_prompt: str,
    actions_desc: str,
    history: List[dict],
    user_input: str,
    rolling_window: int,
    summarize_memory: bool = False,
    prefetch_texts: List[str] | None = None,
) -> List[dict]:
    # Compose system prompt. Avoid duplicating action instructions when the
    # base prompt already includes an "Available actions" section.
    if actions_desc and "available actions" not in system_prompt.lower():
        system_content = system_prompt + "\n\nAvailable actions:\n" + actions_desc
    else:
        system_content = system_prompt


    msgs = [{"role": "system", "content": system_content}]
    # Optionally include a memory summary as a system-level note
    if summarize_memory and history:
        summary = memory_summarizer.summarize(history, rolling_window)
        msgs.append({"role": "system", "content": "Memory summary: " + summary})

    # Optionally include sanitized pre-fetched facts (from pre_send hooks)
    if prefetch_texts:
        combined = "\n\n".join(t for t in prefetch_texts if t)
        if combined:
            msgs.append({"role": "system", "content": "Pre-fetched facts (sanitized):\n" + combined})

    # Include rolling history (already formatted as role/content dicts)
    start = max(0, len(history) - (rolling_window * 2))
    # Ensure message contents are strings when sent to the LLM
    for m in history[start:]:
        content = m.get("content")
        if isinstance(content, dict):
            try:
                content_str = json.dumps(content)
            except Exception:
                content_str = str(content)
        else:
            content_str = str(content)
        msgs.append({"role": m.get("role", "user"), "content": content_str})

    # Add current user input
    msgs.append({"role": "user", "content": user_input})
    return msgs








def try_parse_tool_call(llm_output: str, actions_list: List[actions.ActionSpec]):
    # Be permissive: the model may include extra text around the JSON
    # tool call (or prefixes like model names). Try direct parse first,
    # then search for a JSON object substring.
    try:
        parsed = json.loads(llm_output)
        return actions.validate_tool_call(parsed, actions_list)
    except Exception:
        pass

    # Attempt to find first {...} JSON object in the output
    start = llm_output.find("{")
    end = llm_output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = llm_output[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return actions.validate_tool_call(parsed, actions_list)
    except Exception:
        return None


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
    handlers, pre_send_hooks = load_plugins(settings)

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
    global OLLAMA_SESSION_ENABLED, OLLAMA_SESSION_BOOTSTRAPPED
    if settings.get("ollama_use_session", False):
        if OLLAMA_URL.rstrip("/").endswith("/api/chat"):
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
                                print("Assistant:", end=" ", flush=True)

                        cumulative += part

                        if mode == "text":
                            # print progressively for text responses
                            print(part, end="", flush=True)
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
                        # print the buffered content as it wasn't a tool-call
                        print("Assistant:", end=" ", flush=True)
                        print(cumulative, end="\n", flush=True)
                        final_content = cumulative
                    else:
                        print()  # newline after progressive text
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
        # If the model emitted a `null` action but we have sanitized
        # pre-fetched facts and the user's request looks informational
        # (e.g., weather, definition), prefer returning the sanitized
        # facts as a plain-text assistant reply rather than treating
        # `null` as an executable action.
        informational_keywords = ("weather", "define", "what is", "who is", "tell me", "info", "information")
        if parsed and getattr(parsed, "tool", "") == "null" and prefetch_texts:
            low = (user_input or "").lower()
            if any(k in low for k in informational_keywords):
                # Convert weather snippets into human-friendly sentences
                friendly = []
                for p in prefetch_texts:
                    if isinstance(p, str) and p.lower().startswith("weather (sanitized):"):
                        friendly.append(web_sanitizer.humanize_weather(p))
                    else:
                        friendly.append(p)
                assistant_text = "\n".join(friendly)
                print("Assistant:", assistant_text)
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": assistant_text})
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


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    # Start the main async REPL loop
    main()

