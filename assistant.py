import requests
import json
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma:2b"

SYSTEM_PROMPT = """
You are a home automation assistant.

Available tools:
- get_time
- turn_on_light(room)
- turn_off_light(room)

If a tool is required, respond ONLY in this exact JSON format:
{"tool": "tool_name", "args": {"key": "value"}}

Do not include explanations.
Do not include extra text.
If no tool is needed, respond normally in plain text.
If you can't map a request to a tool, just say you can't do that
"""

def call_llm(user_input):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 150
            }
        }
    )

    data = response.json()

    if "message" not in data:
        raise Exception(f"Unexpected Ollama response: {data}")

    return data["message"]["content"]


def execute_tool(tool_name, args):
    print(tool_name)
    if tool_name == "get_time":
        return f"The time is {datetime.now().strftime('%H:%M:%S')}."

    if tool_name == "turn_on_light":
        room = args.get("room", "unknown")
        return f"Light in {room} turned on."

    if tool_name == "turn_off_light":
        room = args.get("room", "unknown")
        return f"Light in {room} turned off."

    return "Unknown tool."


def main():
    while True:
        user_input = input("You: ")

        llm_output = call_llm(user_input)

        # Try to parse JSON tool call
        try:
            parsed = json.loads(llm_output)
            if "tool" in parsed:
                result = execute_tool(parsed["tool"], parsed.get("args", {}))
                print("Assistant:", result)
                continue
        except json.JSONDecodeError:
            pass

        # If not JSON, treat as normal reply
        print("Assistant:", llm_output)


if __name__ == "__main__":
    main()

