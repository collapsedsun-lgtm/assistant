import argparse
import asyncio
import json
import os
from typing import List

import aiohttp
import inspect

from llm_config import MODEL, OLLAMA_URL, LLM_OPTIONS
import actions
from plugin_loader import load_plugins
import memory_summarizer

# Number of most-recent exchanges (user + assistant) to include in context.
# This is a hardcoded constant for now; make configurable later if needed.
ROLLING_WINDOW = 5


def load_system_prompt() -> str:
    path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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
    if pre_send_hooks:
        for hook in pre_send_hooks:
            try:
                # Try calling the hook with a `debug` flag when supported.
                maybe = None
                try:
                    sig = inspect.signature(hook)
                    if len(sig.parameters) >= 3:
                        maybe = await hook(user_input, history, debug)
                    else:
                        maybe = await hook(user_input, history)
                except (TypeError, ValueError):
                    # Fallback: try simple call signatures
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

    # Increase total timeout to allow slower model responses
    timeout = aiohttp.ClientTimeout(total=120)
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

    return data["message"]["content"], prefetch_texts


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
    timeout = aiohttp.ClientTimeout(total=120)
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
    try:
        parsed = json.loads(llm_output)
    except json.JSONDecodeError:
        return None

    return actions.validate_tool_call(parsed, actions_list)


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
    args = parser.parse_args()

    history: List[dict] = []
    actions_list = actions.load_actions()
    handlers, pre_send_hooks = load_plugins()

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
                assistant_text = "\n".join(prefetch_texts)
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
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": parsed_json})
            continue

        # Not a tool call: normal assistant reply
        print("Assistant:", llm_output)
        # Save the user message then assistant reply in chronological order
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": llm_output})


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

