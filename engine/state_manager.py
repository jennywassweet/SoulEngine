"""
State manager for reading and updating markdown files with YAML frontmatter
"""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Any


class StateManager:
    """Manages state in markdown files with YAML frontmatter"""

    def __init__(self, file_path: Path | str):
        """
        Initialize state manager for a specific file

        Args:
            file_path: Path to markdown file with YAML frontmatter
        """
        self.file_path = Path(file_path)

    def read(self) -> tuple[dict[str, Any], str]:
        """
        Read frontmatter and content from file

        Returns:
            Tuple of (frontmatter_dict, markdown_content)
        """
        if not self.file_path.exists():
            return {}, ""

        content = self.file_path.read_text(encoding="utf-8")

        # Parse YAML frontmatter
        if content.startswith("---\n"):
            parts = content.split("---\n", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1]) or {}
                markdown = parts[2].strip()
                return frontmatter, markdown

        return {}, content.strip()

    def write(self, frontmatter: dict[str, Any], markdown: str = "") -> None:
        """
        Write frontmatter and content to file

        Args:
            frontmatter: Dictionary to serialize as YAML frontmatter
            markdown: Markdown content (optional)
        """
        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize frontmatter
        frontmatter_yaml = yaml.safe_dump(
            frontmatter,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False
        )

        # Combine into markdown with frontmatter
        content = f"---\n{frontmatter_yaml}---\n\n{markdown.strip()}\n"

        self.file_path.write_text(content, encoding="utf-8")

    def update_frontmatter(self, updates: dict[str, Any]) -> None:
        """
        Update specific frontmatter fields without changing content

        Args:
            updates: Dictionary of fields to update
        """
        frontmatter, markdown = self.read()
        frontmatter.update(updates)
        self.write(frontmatter, markdown)

    def get_frontmatter(self) -> dict[str, Any]:
        """Get only the frontmatter"""
        frontmatter, _ = self.read()
        return frontmatter

    def get_content(self) -> str:
        """Get only the markdown content"""
        _, content = self.read()
        return content

    def increment_counter(self, field: str, amount: int = 1) -> int:
        """
        Increment a numeric counter in frontmatter

        Args:
            field: Field name to increment
            amount: Amount to add (default 1)

        Returns:
            New value after increment
        """
        frontmatter, markdown = self.read()
        current = frontmatter.get(field, 0)
        new_value = current + amount
        frontmatter[field] = new_value
        frontmatter["last_updated"] = datetime.now().isoformat()
        self.write(frontmatter, markdown)
        return new_value

    def reset_counter(self, field: str) -> None:
        """
        Reset a counter to 0

        Args:
            field: Field name to reset
        """
        self.update_frontmatter({
            field: 0,
            "last_updated": datetime.now().isoformat()
        })

    def update_content(self, new_content: str) -> None:
        """
        Update only the markdown content, preserving frontmatter

        Args:
            new_content: New markdown content
        """
        frontmatter, _ = self.read()
        frontmatter["last_updated"] = datetime.now().isoformat()
        self.write(frontmatter, new_content)
