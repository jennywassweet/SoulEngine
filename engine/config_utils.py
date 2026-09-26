"""
Shared helpers for reading soul/config.yaml and the per-setting overrides
under soul/settings/<name>/ — used by both engine/main.py (startup) and
engine/command_executor.py (the /switch control command, which needs to
resolve a setting's llm config before it's the active one).
"""

import re
from pathlib import Path

import yaml

CONFIG_RELATIVE_PATH = Path("soul") / "config.yaml"

# Name of the generated-state directory inside a setting. Defined once here
# and imported everywhere else so there is a single answer to "where does
# runtime live" — see setting_dir/runtime_dir below.
RUNTIME_DIR_NAME = "runtime"


def read_config(base_path: Path) -> dict:
    """Load soul/config.yaml fresh from disk."""
    with open(base_path / CONFIG_RELATIVE_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def setting_dir(base_path: Path, setting: str) -> Path:
    """Everything about one setting: soul/settings/<setting>/.

    Authored files (bibles, relationship, the optional prompt/llm/images
    overrides) sit directly here; everything generated sits under runtime/.
    Nothing about a setting lives outside this directory, which is what makes
    deleting a world one `rm -rf` with nothing orphaned behind it."""
    return Path(base_path) / "soul" / "settings" / setting


def runtime_dir(base_path: Path, setting: str) -> Path:
    """Generated state for one setting: soul/settings/<setting>/runtime/.

    Chat log, summaries, retrieval store, photos, session counters and
    continuity notes — everything the app writes and can regenerate. Ignored
    by git as a whole directory rather than by file name, so a new kind of
    runtime file can't quietly become committable.

    Runtime lives *inside* the setting, not in a parallel data/ tree, because
    the parallel tree let the two halves drift apart: deleting a setting left
    its history orphaned, and recreating a setting under the same name then
    silently inherited that history — old messages, id counters and vectors
    included."""
    return setting_dir(base_path, setting) / RUNTIME_DIR_NAME


def list_settings(base_path: Path) -> list[str]:
    """Names of all settings under soul/settings/, sorted."""
    settings_dir = Path(base_path) / "soul" / "settings"
    if not settings_dir.is_dir():
        return []
    return sorted(p.name for p in settings_dir.iterdir() if p.is_dir())


def read_setting_characters(base_path: Path, setting: str) -> dict[str, str]:
    """Read a setting's own `self`/`other` pairing from
    soul/settings/<setting>/characters.yaml.

    The pairing belongs to the setting, not to the global config: switching
    settings while carrying over the previously active `self`/`other` silently
    points PromptBuilder at bible files that don't exist in the new setting,
    and missing bibles load as empty strings rather than failing loudly.

    Returns {} when the file is absent or declares neither key.
    """
    path = setting_dir(base_path, setting) / "characters.yaml"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        declared = yaml.safe_load(f) or {}
    return {key: declared[key] for key in ("self", "other") if declared.get(key)}


def resolve_llm_config(base_path: Path, setting: str, llm_config: dict) -> dict:
    """Per-setting llm.yaml overrides individual keys of the shared llm config,
    same isolation-boundary pattern as main_prompt.md/compression_prompt.md but
    merged key-by-key instead of swapping the whole file — a setting can override
    just `model` without redeclaring temperature/max_tokens/etc."""
    override_path = setting_dir(base_path, setting) / "llm.yaml"
    if not override_path.exists():
        return llm_config
    with open(override_path, "r") as f:
        override = yaml.safe_load(f) or {}
    return {**llm_config, **override}


def resolve_images_config(base_path: Path, setting: str, images_config: dict) -> dict:
    """Per-setting images.yaml declares which photo modes are open here and
    can override individual keys of the shared images config (model, the
    nested llm block), merged key-by-key like resolve_llm_config - a setting
    can override just `model` without redeclaring the rest.

    Absence of images.yaml means photos are off for this setting - `modes`
    comes back empty - regardless of the global images.enabled switch,
    which means "don't spend money", not "appropriate for this pairing"."""
    override_path = setting_dir(base_path, setting) / "images.yaml"
    if not override_path.exists():
        return {**images_config, "modes": []}
    with open(override_path, "r") as f:
        override = yaml.safe_load(f) or {}
    merged = {**images_config, **override}
    merged["llm"] = {**images_config.get("llm", {}), **override.get("llm", {})}
    merged.setdefault("modes", [])
    return merged


def write_active_setting(base_path: Path, setting: str, self_slug: str, other_slug: str) -> None:
    """Rewrite the `setting:`/`characters.self`/`characters.other` lines of
    soul/config.yaml in place via regex substitution (mirrors switch-setting.sh's
    sed approach) so every other line — including the extensive prose comments
    documenting past model choices — survives untouched.

    The pairing is always written alongside the setting: leaving the previous
    setting's slugs in place points PromptBuilder at bibles that don't exist
    under the new setting, which load as empty strings without failing."""
    config_path = base_path / CONFIG_RELATIVE_PATH
    text = config_path.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^setting: .*$", f"setting: {setting}", text, count=1)
    text = re.sub(r"(?m)^  self: .*$", f"  self: {self_slug}", text, count=1)
    text = re.sub(r"(?m)^  other: .*$", f"  other: {other_slug}", text, count=1)
    config_path.write_text(text, encoding="utf-8")
