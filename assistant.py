import argparse
import json
import os
import requests
from datetime import datetime
from typing import List, Dict

from llm_config import MODEL, OLLAMA_URL, LLM_OPTIONS
import actions

# Number of most-recent exchanges (user + assistant) to include in context.
# This is a hardcoded constant for now; make configurable later if needed.
ROLLING_WINDOW = 5


def load_system_prompt() -> str:
    path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_messages(system_prompt: str, actions_desc: str, history: List[Dict], user_input: str) -> List[Dict]:
    # Compose system prompt with a description of available actions
    system_content = system_prompt + "\n\nAvailable actions:\n" + actions_desc

    msgs = [{"role": "system", "content": system_content}]

    # Include rolling history (already formatted as role/content dicts)
    start = max(0, len(history) - (ROLLING_WINDOW * 2))
    msgs.extend(history[start:])

    # Add current user input
    msgs.append({"role": "user", "content": user_input})
    return msgs


def call_llm(user_input: str, history: List[Dict], mock: bool = False) -> str:
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

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=10)
    except requests.RequestException as e:
        raise RuntimeError(f"LLM request failed: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"LLM returned status {resp.status_code}: {resp.text}")

    data = resp.json()
    if "message" not in data or "content" not in data["message"]:
        raise RuntimeError(f"Unexpected LLM response shape: {data}")

    return data["message"]["content"]


def try_parse_tool_call(llm_output: str, actions_list: List[Dict]):
    try:
        parsed = json.loads(llm_output)
    except json.JSONDecodeError:
        return None

    if actions.validate_tool_call(parsed, actions_list):
        return parsed
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use a mock LLM response for testing")
    args = parser.parse_args()

    history: List[Dict] = []
    actions_list = actions.load_actions()

    print("Starting agent REPL (type Ctrl-C to exit).\nActions are only emitted as JSON; this program will not execute them.")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye")
            break

        try:
            llm_output = call_llm(user_input, history, mock=args.mock)
        except Exception as e:
            print("Assistant (error):", e)
            continue

        # Try to parse JSON tool call
        parsed = try_parse_tool_call(llm_output, actions_list)
        if parsed:
            # Do not execute — just print the action JSON so a separate runner can act on it
            print("Assistant (action):", json.dumps(parsed))
            # Save assistant reply (the JSON) in history
            history.append({"role": "assistant", "content": json.dumps(parsed)})
            history.append({"role": "user", "content": user_input})
            continue

        # Not a tool call: normal assistant reply
        print("Assistant:", llm_output)
        history.append({"role": "assistant", "content": llm_output})
        history.append({"role": "user", "content": user_input})


if __name__ == "__main__":
    main()

