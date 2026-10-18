import argparse
import os
import sys
import json
import subprocess

from openai import OpenAI

from app.skills import find_skills, format_skills_for_prompt, get_skill_body, resolve_skill, get_skill_context

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

skill_tool_spec = {
    "type": "function",
    "function": {
        "name": "Skill",
        "description": "Load a skill's instructions into the conversation",
        "parameters": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "description": "The name of the skill to use"},
                "args": {"type": "string", "description": "Optional arguments for the skill"},
            },
        },
    },
}

all_tools = [read_tool_spec, write_tool_spec, bash_tool_spec, skill_tool_spec]


def build_system_prompt():
    skills = find_skills()
    skills_section = format_skills_for_prompt(skills)

    base = "You are a helpful AI coding assistant."

    if skills_section:
        return f"""{base}

You have access to the following skills:

{skills_section}

If a skill matches the user's request, call the Skill tool with its name
and follow the instructions it returns."""

    return base


def run_agent_loop(client, messages):
    """Run the agent loop over the given messages and return the final response text."""
    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=all_tools,
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        messages.append(chat.choices[0].message)

        if chat.choices[0].message.tool_calls is not None:
            tool_call = chat.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments

            if function_name == "Read":
                file_path = json.loads(function_args).get("file_path")
                tool_call_result = open(file_path).read()

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_call_result
                })

            elif function_name == "Write":
                file_path = json.loads(function_args).get("file_path")
                content = json.loads(function_args).get("content")
                with open(file_path, "w") as f:
                    f.write(content)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"File {file_path} written successfully"
                })

            elif function_name == "Bash":
                command = json.loads(function_args).get("command")
                result = subprocess.run(command, capture_output=True, shell=True)

                stdout = result.stdout.decode('utf-8')
                stderr = result.stderr.decode('utf-8')

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": stderr if stderr else stdout
                })

            elif function_name == "Skill":
                skill_name = json.loads(function_args).get("name")
                skill_args_str = json.loads(function_args).get("args", "")
                skill_args = skill_args_str.split() if skill_args_str else None

                skill_path = resolve_skill(skill_name)
                if not skill_path:
                    tool_call_result = f"Skill '{skill_name}' not found"
                else:
                    context = get_skill_context(skill_path)
                    if context == "fork":
                        # Run the skill body in a subagent (separate conversation)
                        body = get_skill_body(skill_path, skill_args)
                        subagent_messages = [{"role": "user", "content": body}]
                        subagent_response = run_agent_loop(client, subagent_messages)
                        tool_call_result = f"Skill {skill_name} ran in a separate context and returned: {subagent_response}"
                    else:
                        tool_call_result = get_skill_body(skill_path, skill_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_call_result
                })

        else:
            return chat.choices[0].message.content


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    system_prompt = build_system_prompt()

    # Handle slash commands: /skill1 /skill2 arg1 arg2
    prompt = args.p
    if prompt.startswith("/"):
        parts = prompt.split()
        skill_names = [p[1:] for p in parts if p.startswith("/")]
        skill_args = [p for p in parts if not p.startswith("/")]

        bodies = []
        for name in skill_names:
            skill_path = resolve_skill(name)
            if skill_path:
                bodies.append(get_skill_body(skill_path, skill_args))

        if bodies:
            prompt = "\n\n".join(bodies)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = run_agent_loop(client, messages)
    print(response)


if __name__ == "__main__":
    main()
