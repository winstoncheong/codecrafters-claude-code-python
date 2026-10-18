import os
from dataclasses import dataclass, field

import yaml

SKILLS_DIR = os.path.join(".claude", "skills")
SKILL_FILE = "SKILL.md"


@dataclass
class Skill:
    name: str
    description: str
    path: str
    context: str | None = None
    body: str = ""

    def expand_body(self, args=None):
        """Return the body with $ARGUMENTS and $0, $1, etc. substituted."""
        result = self.body
        if args:
            result = result.replace("$ARGUMENTS", " ".join(args))
            for i, arg in enumerate(args):
                result = result.replace(f"${i}", arg)
        return result


def parse_frontmatter(content):
    """Parse YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}
    end = content.find("\n---", 3)
    if end == -1:
        return {}
    frontmatter = content[4:end]
    data = yaml.safe_load(frontmatter)
    return data if isinstance(data, dict) else {}


def extract_body(content):
    """Extract the body (everything after the closing '---') from markdown content."""
    end = content.find("\n---", 3)
    if end == -1:
        return content.strip()
    return content[end + 4:].strip()


def find_skills(skills_dir=SKILLS_DIR):
    """Find all skills on disk, reading and parsing each file once."""
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
        body = extract_body(content)

        skills.append(Skill(
            name=metadata.get("name", entry),
            description=metadata.get("description", ""),
            path=skill_file,
            context=metadata.get("context"),
            body=body,
        ))

    return skills


def build_skill_catalog(skills):
    """Build a name-to-Skill lookup dict."""
    return {s.name: s for s in skills}


def format_skills_for_prompt(skills):
    """Format the list of skills for inclusion in the system prompt."""
    if not skills:
        return ""

    lines = []
    for skill in skills:
        line = f"- **{skill.name}**: {skill.description}"
        line += f" (Read `{skill.path}` for full instructions)"
        lines.append(line)

    return "\n".join(lines)
