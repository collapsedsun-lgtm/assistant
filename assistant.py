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
from kv_store import get_default_kv, messages_list_key, response_cache_key
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


async def call_llm(user_input: str, history: List[dict], mock: bool = False, debug: bool = False, rolling_window: int = 5, summarize_memory: bool = False, pre_send_hooks: List[callable] | None = None) -> tuple:
    if mock:
        # Simple mock: if user asks to get time, return a tool call
        if "time" in user_input.lower():
            return json.dumps({"tool": "get_time", "args": {}}), []
        return "I would do that if I could.", []

    system_prompt = load_system_prompt()
    actions_list = actions.load_actions()
    actions_desc = actions.actions_description(actions_list)

    # If Ollama session mode has been enabled and probed, send a short
    # system hint and compact action list to avoid resending the large
    # `system_prompt` and full actions table on every request.
    global OLLAMA_SESSION_ENABLED, OLLAMA_SESSION_BOOTSTRAPPED
    settings = load_settings()
    is_chat_endpoint = OLLAMA_URL.rstrip("/").endswith("/api/chat")
    use_session = settings.get("ollama_use_session", False) and OLLAMA_SESSION_ENABLED and OLLAMA_SESSION_BOOTSTRAPPED and (not is_chat_endpoint)
    if debug:
        try:
            print(
                f"[debug] use_session={use_session} "
                f"(ollama_use_session={settings.get('ollama_use_session', None)}, "
                f"OLLAMA_SESSION_ENABLED={OLLAMA_SESSION_ENABLED}, "
                f"OLLAMA_SESSION_BOOTSTRAPPED={OLLAMA_SESSION_BOOTSTRAPPED}, "
                f"chat_endpoint={is_chat_endpoint})"
            )
        except Exception:
            pass
    if use_session:
        # Session mode relies on bootstrap state; avoid sending additional
        # per-request system hints that can conflict with model behavior.
        system_prompt_to_send = ""
        actions_desc_to_send = ""
    else:
        system_prompt_to_send = system_prompt
        actions_desc_to_send = actions_desc
    if debug and not use_session:
        try:
            print(f"[debug] system_prompt_to_send length: {len(system_prompt_to_send)} chars, actions_desc_to_send length: {len(actions_desc_to_send)}")
        except Exception:
            pass

    # Run pre-send hooks (they should fetch and sanitize web data before returning strings)
    prefetch_texts: List[str] = []
    settings = load_settings()
    t_start = time.perf_counter()
    if pre_send_hooks:
        for hook in pre_send_hooks:
            try:
                maybe = None
                try:
                    sig = inspect.signature(hook)
                    # Build kwargs: pass debug and settings when supported by the hook
                    kwargs = {}
                    params = sig.parameters
                    if "debug" in params:
                        kwargs["debug"] = debug
                    if "settings" in params or "config" in params:
                        kwargs["settings"] = settings

                    maybe = await hook(user_input, history, **kwargs)
                except Exception:
                    # Fallback to simpler call signatures
                    try:
                        maybe = await hook(user_input, history)
                    except TypeError:
                        try:
                            maybe = await hook(user_input)
                        except TypeError:
                            maybe = None

                if maybe:
                    # only accept str results
                    if isinstance(maybe, str):
                        prefetch_texts.append(maybe)
                    else:
                        # If hook returned a mapping, convert to string
                        try:
                            prefetch_texts.append(json.dumps(maybe))
                        except Exception:
                            prefetch_texts.append(str(maybe))
                if debug:
                    print(f"[debug] pre_send hook {getattr(hook, '__name__', repr(hook))} returned: {repr(maybe)[:400]}")
            except Exception:
                # Do not let a failing hook leak raw results or crash the request
                if debug:
                    import traceback

                    print("[debug] pre_send hook failed:\n", traceback.format_exc())
    else:
        if debug:
            print("[debug] no pre_send hooks registered")
    t_after_hooks = time.perf_counter()
    if debug:
        print(f"[debug] pre_send hooks duration: {t_after_hooks - t_start:.3f}s")

    if use_session:
        # In session mode, avoid sending full local memory/history. Keep only
        # recent assistant tool-call replies and the current user turn.
        messages: List[dict] = []

        if prefetch_texts:
            combined = "\n\n".join(t for t in prefetch_texts if t)
            if combined:
                messages.append({"role": "system", "content": "Pre-fetched facts (sanitized):\n" + combined})

        tool_reply_history: List[dict] = []
        for item in reversed(history):
            if item.get("role") != "assistant":
                continue
            content = item.get("content")
            if not isinstance(content, str):
                continue
            try:
                parsed = json.loads(content)
            except Exception:
                continue
            if isinstance(parsed, dict) and "tool" in parsed and "args" in parsed:
                tool_reply_history.append({"role": "assistant", "content": content})
            if len(tool_reply_history) >= 4:
                break
        tool_reply_history.reverse()
        messages.extend(tool_reply_history)
        messages.append({"role": "user", "content": user_input})
    else:
        messages = build_messages(
            system_prompt_to_send,
            actions_desc_to_send,
            history,
            user_input,
            rolling_window,
            summarize_memory,
            prefetch_texts=prefetch_texts,
        )

    payload = {"model": MODEL, "messages": messages}
    payload.update(LLM_OPTIONS)
    if "keep_alive" not in payload:
        payload["keep_alive"] = settings.get("ollama_keep_alive", "30m")

    if debug:
        try:
            print("[debug] LLM payload:", json.dumps(payload, indent=2))
        except Exception:
            print("[debug] (could not serialize payload)")
        print("Please wait, thinking...")

    # Try to read from response cache first
    try:
        kv = await get_default_kv()
    except Exception:
        kv = None

    t_kv_start = time.perf_counter()
    messages_json = None
    cache_key = None
    if kv is not None:
        try:
            messages_json = json.dumps(messages, sort_keys=True)
            cache_key = response_cache_key(MODEL, LLM_OPTIONS, messages_json)
            cached = await kv.get(cache_key)
            if cached is not None:
                if debug:
                    print("[debug] cache hit for request")
                return cached, prefetch_texts

            # If a partial response exists, return it immediately and
            # refresh the final value in background so future requests
            # get the completed response.
            partial_key = cache_key + ":partial"
            partial = await kv.get(partial_key)
            if partial is not None:
                if debug:
                    print("[debug] partial cache hit; attempting to refresh final result synchronously")

                async def _background_refresh(k: "KVStore", ckey: str, payload_obj: dict):
                    # Do a non-streaming request to get the final content and cache it.
                    timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT)
                    try:
                        if debug:
                            try:
                                print("[debug] background refresh payload:", json.dumps(payload_obj)[:1000])
                            except Exception:
                                print("[debug] background refresh payload: (unserializable)")
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            async with session.post(OLLAMA_URL, json=payload_obj) as resp:
                                if resp.status != 200:
                                    text = await resp.text()
                                    if debug:
                                        print(f"[debug] background refresh failed status={resp.status} body={text[:1000]}")
                                    return None
                                data = await resp.json()
                                if debug:
                                    try:
                                        print("[debug] background refresh response:", json.dumps(data)[:1000])
                                    except Exception:
                                        print("[debug] background refresh response: (unserializable)")
                                if "message" in data and "content" in data["message"]:
                                    final_text = data["message"]["content"]
                                    try:
                                        await k.set(ckey, final_text, ex=FINAL_TTL)
                                        await k.delete(ckey + ":partial")
                                    except Exception:
                                        pass
                                    return final_text
                    except Exception:
                        return None

                # Try to refresh synchronously so the first requester gets the final reply
                try:
                    final_now = await _background_refresh(kv, cache_key, payload)
                    if final_now is not None:
                        if debug:
                            print("[debug] refreshed final result from partial cache")
                        return final_now, prefetch_texts
                except Exception:
                    final_now = None

                # If synchronous refresh failed, schedule in background and return partial as fallback
                if debug:
                    print("[debug] synchronous refresh failed; returning partial and scheduling background refresh")
                try:
                    asyncio.create_task(_background_refresh(kv, cache_key, payload))
                except Exception:
                    pass

                return partial, prefetch_texts
        except Exception:
            # If any error occurs during KV/cache checks, continue without cache
            pass
    t_kv_end = time.perf_counter()
    if debug:
        print(f"[debug] KV/cache handling duration: {t_kv_end - t_kv_start:.3f}s")
    

    # Determine whether to use streaming: env toggle or options flag
    use_stream = LLM_STREAM or bool(LLM_OPTIONS.get("stream"))

    if use_stream:
        async def _stream_gen():
            # Buffer-based SSE-style parser: accumulate bytes, split on
            # double-newline between events, extract 'data:' lines that
            # typically contain JSON, parse them and yield textual pieces.
            timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT)
            if debug:
                t_req0 = time.perf_counter()
                print("[debug] starting streaming LLM request")
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.post(OLLAMA_URL, json=payload) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise RuntimeError(f"LLM returned status {resp.status}: {text}")

                        buf = ""
                        assembled = []
                        last_saved_len = 0
                        partial_key = (cache_key + ":partial") if cache_key else None
                        save_threshold = PARTIAL_SAVE_THRESHOLD  # characters between partial writes

                        async for raw_chunk in resp.content.iter_chunked(1024):
                            try:
                                s = raw_chunk.decode()
                            except Exception:
                                s = raw_chunk.decode(errors="ignore")
                            buf += s
                            # Process complete SSE events separated by \n\n
                            while "\n\n" in buf:
                                event, buf = buf.split("\n\n", 1)
                                # Collect `data:` lines
                                data_lines = []
                                for line in event.splitlines():
                                    if line.startswith("data:"):
                                        data_lines.append(line[5:].lstrip())
                                if not data_lines:
                                    # Not an SSE event we care about
                                    continue
                                data_str = "\n".join(data_lines)
                                # Try to parse JSON payloads, otherwise yield raw data
                                try:
                                    obj = json.loads(data_str)
                                except Exception:
                                    # Not JSON: yield as-is
                                    if data_str.strip():
                                        assembled.append(data_str)
                                        yield data_str
                                    continue

                                # Extract textual pieces from JSON delta
                                pieces = _extract_texts(obj)
                                if pieces:
                                    for piece in pieces:
                                        if piece:
                                            assembled.append(piece)
                                            yield piece

                            # Periodically persist partial assembled content
                            if partial_key and assembled and len("".join(assembled)) - last_saved_len >= save_threshold:
                                try:
                                    await kv.set(partial_key, "".join(assembled), ex=PARTIAL_TTL)
                                    last_saved_len = len("".join(assembled))
                                except Exception:
                                    pass

                        # Flush any remaining buffered text when stream ends
                        if buf.strip():
                            try:
                                obj = json.loads(buf)
                                pieces = _extract_texts(obj)
                                for piece in pieces:
                                    if piece:
                                        assembled.append(piece)
                                        yield piece
                            except Exception:
                                assembled.append(buf)
                                yield buf

                        # Finalize: cache final assembled response and remove partial
                        final_text = "".join(assembled)
                        if kv is not None and cache_key:
                            try:
                                await kv.set(cache_key, final_text, ex=FINAL_TTL)
                                if partial_key:
                                    await kv.delete(partial_key)
                            except Exception:
                                pass
                        if debug:
                            t_req1 = time.perf_counter()
                            print(f"[debug] streaming LLM total duration: {t_req1 - t_req0:.3f}s")
                except Exception as e:
                    import traceback
                    raise RuntimeError(f"LLM streaming request failed: {e!r}\n{traceback.format_exc()}")

        return _stream_gen(), prefetch_texts

    # Non-streaming (legacy) path: request full JSON and cache result
    timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(OLLAMA_URL, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"LLM returned status {resp.status}: {text}")
                data = await resp.json()
        except Exception as e:
            import traceback
            raise RuntimeError(f"LLM request failed: {e!r}\n{traceback.format_exc()}")

    if "message" not in data or "content" not in data["message"]:
        raise RuntimeError(f"Unexpected LLM response shape: {data}")

    final = data["message"]["content"]
    if kv is not None and cache_key:
        try:
            await kv.set(cache_key, final, ex=FINAL_TTL)
        except Exception:
            pass

    return final, prefetch_texts


async def check_model_endpoint(debug: bool = False):
    """Connectivity-only check for the Ollama server.

    Returns (success: bool, info: str) by requesting the tags endpoint on the
    same host as `OLLAMA_URL`, without sending model prompts.
    """
    import traceback
    from urllib.parse import urlparse

    parsed = urlparse(OLLAMA_URL)
    if not parsed.scheme or not parsed.netloc:
        return False, f"invalid OLLAMA_URL: {OLLAMA_URL}"
    tags_url = f"{parsed.scheme}://{parsed.netloc}/api/tags"

    if debug:
        print(f"[debug] connectivity check URL: {tags_url}")

    timeout = aiohttp.ClientTimeout(total=min(15, LLM_TIMEOUT))
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(tags_url) as resp:
                text = await resp.text()
                if resp.status != 200:
                    return False, f"status={resp.status} body={text[:500]}"
                try:
                    data = await resp.json()
                except Exception as e:
                    return False, f"invalid json response: {e} / body={text[:500]}"
    except Exception as e:
        tb = traceback.format_exc()
        return False, tb

    if not isinstance(data, dict) or "models" not in data:
        return False, f"unexpected tags response shape: {data}"

    return True, f"connected ({len(data.get('models', []))} models listed)"


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
    kv = await get_default_kv()
    try:
        raw = await kv.lrange(messages_list_key(session_id), 0, -1)
        if raw:
            for item in raw:
                try:
                    history.append(json.loads(item))
                except Exception:
                    history.append({"role": "assistant", "content": item})
    except Exception:
        # If KV is unavailable, continue with in-memory history only
        pass
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
            await kv.rpush(messages_list_key(session_id), json.dumps({"role": "user", "content": user_input}))
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

                            # Print and persist the formatted assistant reply
                            print("Assistant:", formatted_text)
                            history.append({"role": "assistant", "content": formatted_text})
                            try:
                                await kv.rpush(messages_list_key(session_id), json.dumps({"role": "assistant", "content": formatted_text}))
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
                await kv.rpush(messages_list_key(session_id), json.dumps({"role": "assistant", "content": parsed_json}))
            except Exception:
                pass
            continue

        # Not a tool call: normal assistant reply
        print("Assistant:", llm_output)
        # Save the user message then assistant reply in chronological order
        history.append({"role": "assistant", "content": llm_output})
        try:
            await kv.rpush(messages_list_key(session_id), json.dumps({"role": "assistant", "content": llm_output}))
        except Exception:
            pass


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    # Start the main async REPL loop
    main()

