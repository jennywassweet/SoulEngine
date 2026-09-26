"""
Debug logger - saves prompts to files for debugging
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


# Default role-to-name mapping
DEFAULT_ROLE_NAMES = {
    "user": "USER",
    "assistant": "ASSISTANT",
    "system": "SYSTEM",
}


def save_prompt_to_file(
    messages: list[dict[str, Any]],
    file_path: Path | str,
    role_names: dict[str, str] | None = None,
) -> None:
    """
    Save prompt messages to a debug file

    Args:
        messages: List of message dictionaries (role, content)
        file_path: Path to save the prompt
        role_names: Optional mapping of role -> display name
                    (e.g. {"user": "Misha", "assistant": "Zoe"})
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    names = {**DEFAULT_ROLE_NAMES, **(role_names or {})}

    # Format messages for readability
    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        display_name = names.get(role, role.upper())
        formatted.append(f"========== {display_name} ==========")
        formatted.append(content)
        formatted.append("")

    # Write to file
    file_path.write_text("\n".join(formatted), encoding="utf-8")


def append_debug_log(record: dict[str, Any], log_path: Path | str) -> None:
    """
    Append one full record of a model call (prompt in, response out) to an
    append-only JSONL log — unlike save_prompt_to_file, nothing is overwritten.
    Meant for reviewing a whole test session later (e.g. judging how well
    compression preserves continuity across many calls).

    Args:
        record: Arbitrary JSON-serializable dict describing the call
                (e.g. type, active characters, messages, parsed response)
        log_path: Path to the .jsonl log file
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {"timestamp": datetime.now().isoformat(), **record}

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
