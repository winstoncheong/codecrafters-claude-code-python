import argparse
import os
import sys
import json
import subprocess

from openai import OpenAI

from app.skills import find_skills, format_skills_for_prompt

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

write_tool_spec = {
    "type": "function",
    "function": {
        "name": "Write",
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "required": ["file_path", "content"],
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path of the file to write to",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
        },
    },
}

read_tool_spec = {
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

bash_tool_spec = {
    "type": "function",
    "function": {
        "name": "Bash",
        "description": "Execute a shell command",
        "parameters": {
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {"type": "string", "description": "The command to execute"}
            },
        },
    },
}

def build_system_prompt():
    skills = find_skills()
    skills_section = format_skills_for_prompt(skills)

    base = "You are a helpful AI coding assistant."

    if skills_section:
        return f"""{base}

The following skills are available. When a user's request matches a skill's
description, use the Read tool to load the skill's full instructions and
follow them.

{skills_section}"""

    return base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    system_prompt = build_system_prompt()
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.p},
    ]

    while True: # agent loop

        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=[ read_tool_spec, write_tool_spec , bash_tool_spec ],
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        # print(f"#choices: {len(chat.choices)}")

        # add messages to the list regardless
        messages.append(chat.choices[0].message)

        if chat.choices[0].message.tool_calls is not None: # perform a tool call
            tool_call = chat.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments

            # print(f"Tool call: {function_name} with args: {function_args}")

            if function_name == "Read": # execute the Read tool call
                file_path = json.loads(function_args).get("file_path")
                tool_call_result = open(file_path).read()

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_call_result
                })

            elif function_name == "Write": # execute the Write tool call
                file_path = json.loads(function_args).get("file_path")
                content = json.loads(function_args).get("content")
                with open(file_path, "w") as f:
                    f.write(content)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"File {file_path} written successfully"
                })
            
            elif function_name == "Bash": # execute the Bash tool call
                command = json.loads(function_args).get("command")
                result = subprocess.run(command, capture_output=True, shell=True)

                stdout = result.stdout.decode('utf-8')
                stderr = result.stderr.decode('utf-8')

                # print(f'Bash command stdout: {stdout}')
                # print(f'Bash command stderr: {stderr}')

                message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": stderr if stderr else stdout
                }

                # print('Appending to messages:', message)

                messages.append(message)
            
        else: # No tool_calls array
            print(chat.choices[0].message.content)
            break


    # You can use print statements as follows for debugging, they'll be visible when running tests.
    # print("Logs from your program will appear here!", file=sys.stderr)


if __name__ == "__main__":
    main()
