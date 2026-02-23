import argparse
import asyncio
import json
import os
from typing import List

import aiohttp

from llm_config import MODEL, OLLAMA_URL, LLM_OPTIONS
import actions
from plugin_loader import load_plugins

# Number of most-recent exchanges (user + assistant) to include in context.
# This is a hardcoded constant for now; make configurable later if needed.
ROLLING_WINDOW = 5


def load_system_prompt() -> str:
    path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_messages(system_prompt: str, actions_desc: str, history: List[dict], user_input: str) -> List[dict]:
    # Compose system prompt with a description of available actions
    system_content = system_prompt + "\n\nAvailable actions:\n" + actions_desc

    msgs = [{"role": "system", "content": system_content}]

    # Include rolling history (already formatted as role/content dicts)
    start = max(0, len(history) - (ROLLING_WINDOW * 2))
    msgs.extend(history[start:])

    # Add current user input
    msgs.append({"role": "user", "content": user_input})
    return msgs


async def call_llm(user_input: str, history: List[dict], mock: bool = False) -> str:
    if mock:
        # Simple mock: if user asks to get time, return a tool call
        if "time" in user_input.lower():
            return json.dumps({"tool": "get_time", "args": {}})
        return "I would do that if I could."

    system_prompt = load_system_prompt()
    actions_list = actions.load_actions()
    actions_desc = actions.actions_description(actions_list)

    messages = build_messages(system_prompt, actions_desc, history, user_input)

    payload = {"model": MODEL, "messages": messages}
    payload.update(LLM_OPTIONS)

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(OLLAMA_URL, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"LLM returned status {resp.status}: {text}")
                data = await resp.json()
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}")

    if "message" not in data or "content" not in data["message"]:
        raise RuntimeError(f"Unexpected LLM response shape: {data}")

    return data["message"]["content"]


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
    args = parser.parse_args()

    history: List[dict] = []
    actions_list = actions.load_actions()
    plugins = load_plugins()

    print("Starting async agent REPL (type Ctrl-C to exit).\nActions are only emitted as JSON; this program will not execute them.")

    while True:
        try:
            user_input = await ainput("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye")
            break

        try:
            llm_output = await call_llm(user_input, history, mock=args.mock)
        except Exception as e:
            print("Assistant (error):", e)
            continue

        parsed = try_parse_tool_call(llm_output, actions_list)
        if parsed:
            # Do not execute plugins here by default; just show the action JSON
            print("Assistant (action):", parsed.json())
            # Save assistant reply (the JSON) in history
            history.append({"role": "assistant", "content": parsed.json()})
            history.append({"role": "user", "content": user_input})
            continue

        # Not a tool call: normal assistant reply
        print("Assistant:", llm_output)
        history.append({"role": "assistant", "content": llm_output})
        history.append({"role": "user", "content": user_input})


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

