import os

import yaml

SKILLS_DIR = os.path.join(".claude", "skills")
SKILL_FILE = "SKILL.md"


def parse_frontmatter(content):
    """Parse YAML frontmatter from markdown content.

    Frontmatter is delimited by '---' on the first line and '---' on its
    own line. Returns a dict of metadata, or an empty dict if none found.
    """
    if not content.startswith("---"):
        return {}

    # Find the closing '---' delimiter
    end = content.find("\n---", 3)
    if end == -1:
        return {}

    frontmatter = content[4:end]  # skip opening '---\n'
    data = yaml.safe_load(frontmatter)
    return data if isinstance(data, dict) else {}


def find_skills(skills_dir=SKILLS_DIR):
    """Find all skills on disk under the skills directory.

    Each skill is a subdirectory containing a SKILL.md file with YAML
    frontmatter (name, description). Returns a list of skill dicts sorted
    by name.
    """
    skills = []

    if not os.path.isdir(skills_dir):
        return skills

    for entry in sorted(os.listdir(skills_dir)):
        skill_dir = os.path.join(skills_dir, entry)
        skill_file = os.path.join(skill_dir, SKILL_FILE)

        if not os.path.isdir(skill_dir) or not os.path.isfile(skill_file):
            continue

        with open(skill_file, encoding="utf-8") as f:
            content = f.read()

        metadata = parse_frontmatter(content)
        name = metadata.get("name", entry)
        description = metadata.get("description", "")

        skills.append({
            "name": name,
            "description": description,
            "path": skill_file,
        })

    return skills


def format_skills_for_prompt(skills):
    """Format the list of skills for inclusion in the system prompt."""
    if not skills:
        return ""

    lines = []
    for skill in skills:
        line = f"- **{skill['name']}**: {skill['description']}"
        line += f" (Read `{skill['path']}` for full instructions)"
        lines.append(line)

    return "\n".join(lines)
