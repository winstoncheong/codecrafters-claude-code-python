import argparse
import os
import sys
import json

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages=[{"role": "user", "content": args.p}]

    while True: # agent loop

        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "description": "Read and return the contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "The path to the file to read",
                                }
                            },
                            "required": ["file_path"],
                        },
                    },
                }
            ],
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        # add messages to the list regardless
        messages.append(chat.choices[0].message)

        if chat.choices[0].message.tool_calls is not None: # perform a tool call
            tool_call = chat.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments

            if function_name == "Read": # execute the Read tool call
                file_path = json.loads(function_args).get("file_path")
                tool_call_result = open(file_path).read()

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_call_result
                })
        else: # No tool_calls array
            print(chat.choices[0].message.content)
            break


    # You can use print statements as follows for debugging, they'll be visible when running tests.
    # print("Logs from your program will appear here!", file=sys.stderr)


if __name__ == "__main__":
    main()
