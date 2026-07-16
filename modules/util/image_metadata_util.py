import re


_ANGLE_BRACKET_SEGMENT_PATTERN = re.compile(r"<[^<>]*>")


def strip_angle_bracket_segments(prompt: str) -> str:
    """Remove metadata-like ``<...>`` segments from a caption or prompt."""
    cleaned = _ANGLE_BRACKET_SEGMENT_PATTERN.sub("", prompt or "")
    cleaned = re.sub(r"\s*\n\s*", ", ", cleaned)
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"^\s*,\s*|\s*,\s*$", "", cleaned)
    return cleaned.strip()
