"""
Utility for parsing LLM responses with markers
"""


def parse_markers(text: str, markers: list[str]) -> dict[str, str | None]:
    """
    Parse text with markers into dictionary

    Args:
        text: Text containing markers (e.g., "===STATE===")
        markers: List of markers to find (e.g., ["===STATE===", "===USER_MODEL==="])

    Returns:
        Dictionary where key is marker name (without ===), value is text between markers
        If marker not found, value is None

    Example:
        >>> text = "===STATE===\\nstate text\\n===USER_MODEL===\\nuser model text"
        >>> parse_markers(text, ["===STATE===", "===USER_MODEL==="])
        {'STATE': 'state text', 'USER_MODEL': 'user model text'}
    """
    result = {}

    for i, marker in enumerate(markers):
        # Extract marker name (remove ===)
        marker_name = marker.replace("===", "").strip()

        # Find marker position — last occurrence, not first. A model that
        # narrates its own instructions before committing to the real
        # answer (e.g. reasoning prose that quotes "===LINE===" while
        # describing the format) can make the marker appear earlier than
        # its real, structural use; the genuine one is always the last.
        marker_pos = text.rfind(marker)

        if marker_pos == -1:
            # Marker not found
            result[marker_name] = None
            continue

        # Start position after marker
        start = marker_pos + len(marker)

        # Find next marker or end of text
        if i + 1 < len(markers):
            next_marker = markers[i + 1]
            next_pos = text.find(next_marker, start)
            if next_pos != -1:
                end = next_pos
            else:
                end = len(text)
        else:
            end = len(text)

        # Extract and strip content
        content = text[start:end].strip()
        result[marker_name] = content

    return result
