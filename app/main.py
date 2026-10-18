import argparse
import os
import sys
import json
import subprocess

from openai import OpenAI

from app.skills import find_skills, build_skill_catalog, format_skills_for_prompt

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


def handle_read(args):
    file_path = args.get("file_path")
    return open(file_path).read()


def handle_write(args):
    file_path = args.get("file_path")
    content = args.get("content")
    with open(file_path, "w") as f:
        f.write(content)
    return f"File {file_path} written successfully"


def handle_bash(args):
    command = args.get("command")
    result = subprocess.run(command, capture_output=True, shell=True)
    stdout = result.stdout.decode("utf-8")
    stderr = result.stderr.decode("utf-8")
    return stderr if stderr else stdout


def handle_skill(args, client, catalog):
    skill_name = args.get("name")
    args_str = args.get("args", "")
    skill_args = args_str.split() if args_str else None

    skill = catalog.get(skill_name)
    if not skill:
        return f"Skill '{skill_name}' not found"

    if skill.context == "fork":
        body = skill.expand_body(skill_args)
        subagent_messages = [{"role": "user", "content": body}]
        subagent_response = run_agent_loop(client, subagent_messages)
        return f"Skill {skill_name} ran in a separate context and returned: {subagent_response}"

    return skill.expand_body(skill_args)


def run_agent_loop(client, messages, catalog):
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
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "Read":
                result = handle_read(function_args)
            elif function_name == "Write":
                result = handle_write(function_args)
            elif function_name == "Bash":
                result = handle_bash(function_args)
            elif function_name == "Skill":
                result = handle_skill(function_args, client, catalog)
            else:
                result = f"Unknown tool: {function_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
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

    skills = find_skills()
    catalog = build_skill_catalog(skills)
    system_prompt = build_system_prompt()

    # Handle slash commands: /skill1 /skill2 arg1 arg2
    prompt = args.p
    if prompt.startswith("/"):
        parts = prompt.split()
        skill_names = [p[1:] for p in parts if p.startswith("/")]
        skill_args = [p for p in parts if not p.startswith("/")]

        bodies = []
        for name in skill_names:
            skill = catalog.get(name)
            if skill:
                bodies.append(skill.expand_body(skill_args))

        if bodies:
            prompt = "\n\n".join(bodies)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = run_agent_loop(client, messages, catalog)
    print(response)


if __name__ == "__main__":
    main()
