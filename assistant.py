import argparse
import asyncio
import json
import os
from typing import List

import aiohttp
import inspect

from llm_config import MODEL, OLLAMA_URL, LLM_OPTIONS
from llm_config import LLM_STREAM, PARTIAL_TTL, FINAL_TTL, PARTIAL_SAVE_THRESHOLD, LLM_TIMEOUT
import actions
from plugin_loader import load_plugins
import memory_summarizer
import web_sanitizer
from kv_store import get_default_kv, messages_list_key, response_cache_key
from llm_config import LLM_STREAM


def _extract_texts(obj):
    """Recursively collect string pieces from JSON objects.

    This is permissive: it gathers any string leaf values so we can
    handle different streaming JSON shapes produced by various LLM
    servers (e.g., keys named `content`, `delta`, `text`, or nested
    message structures).
    """
    out = []
    if obj is None:
        return out
    if isinstance(obj, str):
        out.append(obj)
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
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
    # Compose system prompt with a description of available actions
    system_content = system_prompt + "\n\nAvailable actions:\n" + actions_desc


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

    # Run pre-send hooks (they should fetch and sanitize web data before returning strings)
    prefetch_texts: List[str] = []
    settings = load_settings()
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

    messages = build_messages(system_prompt, actions_desc, history, user_input, rolling_window, summarize_memory, prefetch_texts=prefetch_texts)

    payload = {"model": MODEL, "messages": messages}
    payload.update(LLM_OPTIONS)

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
                    print("[debug] partial cache hit; returning partial and refreshing in background")

                async def _background_refresh(p: dict, k: KVStore, ckey: str, payload_obj: dict):
                    # Do a non-streaming request to get the final content and cache it.
                    timeout = aiohttp.ClientTimeout(total=120)
                    try:
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            async with session.post(OLLAMA_URL, json=payload_obj) as resp:
                                if resp.status != 200:
                                    return
                                data = await resp.json()
                                if "message" in data and "content" in data["message"]:
                                    final_text = data["message"]["content"]
                                    try:
                                        await k.set(ckey, final_text, ex=FINAL_TTL)
                                        await k.delete(ckey + ":partial")
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                # schedule background refresh
                try:
                    asyncio.create_task(_background_refresh({}, kv, cache_key, payload))
                except Exception:
                    pass

                return partial, prefetch_texts
        except Exception:
            pass

    # Determine whether to use streaming: env toggle or options flag
    use_stream = LLM_STREAM or bool(LLM_OPTIONS.get("stream"))

    if use_stream:
        async def _stream_gen():
            # Buffer-based SSE-style parser: accumulate bytes, split on
            # double-newline between events, extract 'data:' lines that
            # typically contain JSON, parse them and yield textual pieces.
            timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT)
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
            await kv.set(cache_key, final, ex=60)
        except Exception:
            pass

    return final, prefetch_texts


async def check_model_endpoint(debug: bool = False):
    """Send a small ping to the LLM endpoint and return (success: bool, info: str).

    If `debug` is True, the JSON payload and detailed exception traceback will be printed/returned.
    """
    import traceback

    # Use a narrow health-check system message that asks for a plain-text
    # response. This avoids the main `system_prompt` which forces the
    # model to emit JSON tool calls (and therefore return `null`).
    health_system = "Health-check: ignore any tool-call instructions and respond with plain text 'pong' or a short status message. Do NOT emit JSON."
    test_messages = [{"role": "system", "content": health_system}, {"role": "user", "content": "ping"}]
    payload = {"model": MODEL, "messages": test_messages}
    payload.update(LLM_OPTIONS)

    if debug:
        try:
            print("[debug] LLM health-check payload:", json.dumps(payload, indent=2))
        except Exception:
            print("[debug] (could not serialize health-check payload)")
        print("Please wait, thinking...")

    # Health check may take longer depending on model load; allow more time
    timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(OLLAMA_URL, json=payload) as resp:
                text = await resp.text()
                if resp.status != 200:
                    return False, f"status={resp.status} body={text}"
                try:
                    data = await resp.json()
                except Exception as e:
                    return False, f"invalid json response: {e} / body={text}"
    except Exception as e:
        tb = traceback.format_exc()
        return False, tb

    if "message" not in data or "content" not in data["message"]:
        return False, f"unexpected response shape: {data}"

    return True, data["message"]["content"]


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

    # Print available actions and plugins for visibility/debugging
    actions_desc = actions.actions_description(actions_list)
    print("Starting async agent REPL (type Ctrl-C to exit).\nActions are only emitted as JSON; this program will not execute them.")
    print("\nAvailable actions:")
    print(actions_desc)

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
            print("\nLLM check OK. Sample response:", info)
        else:
            print("\nLLM check failed:", info)
            ans = await ainput("Continue without LLM (you can use --mock)? (y/N): ")
            if ans.strip().lower() not in ("y", "yes"):
                print("Exiting due to failed LLM health check.")
                return
    else:
        print("\nSkipping LLM health check because --mock was provided.")

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
    main()

