#!/usr/bin/env bash
# Switch the active setting in soul/config.yaml. The self/other pairing is not
# passed in — it's read from the setting's own characters.yaml, so it always
# matches the setting being switched to. Edits only the relevant lines via sed,
# so comments and formatting in the file are preserved.
#
# Usage:
#   ./switch-setting.sh <setting>
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="soul/config.yaml"
SETTINGS_DIR="soul/settings"

list_settings() {
    echo "Available settings:"
    for d in "$SETTINGS_DIR"/*/; do
        [ -d "$d" ] && echo "  - $(basename "$d")"
    done
}

if [ $# -lt 1 ]; then
    echo "Usage: ./switch-setting.sh <setting>"
    list_settings
    exit 1
fi

SETTING="$1"
SETTING_PATH="$SETTINGS_DIR/$SETTING"

if [ ! -d "$SETTING_PATH" ]; then
    echo "Error: $SETTING_PATH does not exist."
    list_settings
    exit 1
fi

CHARACTERS_FILE="$SETTING_PATH/characters.yaml"
if [ ! -f "$CHARACTERS_FILE" ]; then
    echo "Error: $CHARACTERS_FILE does not exist — it must declare self/other."
    exit 1
fi

SELF=$(sed -n 's/^self: *//p' "$CHARACTERS_FILE" | head -1)
OTHER=$(sed -n 's/^other: *//p' "$CHARACTERS_FILE" | head -1)

if [ -z "$SELF" ] || [ -z "$OTHER" ]; then
    echo "Error: $CHARACTERS_FILE must declare both 'self:' and 'other:'."
    exit 1
fi

for slug in "$SELF" "$OTHER"; do
    if [ ! -d "$SETTING_PATH/characters/$slug" ]; then
        echo "Error: $CHARACTERS_FILE declares '$slug', but $SETTING_PATH/characters/$slug/ does not exist."
        exit 1
    fi
done

sed -i '' "s/^setting: .*/setting: $SETTING/" "$CONFIG"
sed -i '' "s/^  self: .*/  self: $SELF/" "$CONFIG"
sed -i '' "s/^  other: .*/  other: $OTHER/" "$CONFIG"

echo "Now active:"
grep "^setting:" "$CONFIG"
grep -A2 "^characters:" "$CONFIG"
