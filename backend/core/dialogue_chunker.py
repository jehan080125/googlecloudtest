import re

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…。！？])\s+")
CLAUSE_BOUNDARY_RE = re.compile(r"(?<=[,，;:])\s+")


def split_dialogue_chunks(
    text: str,
    *,
    max_chars: int = 160,
    max_chunks: int = 8,
) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    if max_chars <= 0:
        return [cleaned]

    paragraphs = [part.strip() for part in re.split(r"(?:\r?\n)+", cleaned) if part and part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        if not normalized:
            continue
        units.extend(_split_by_sentence(normalized))

    chunks = _pack_units(units, max_chars=max_chars, max_chunks=max_chunks)
    return chunks if chunks else [cleaned]


def _split_by_sentence(text: str) -> list[str]:
    pieces = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part and part.strip()]
    return pieces if pieces else [text.strip()]


def _pack_units(units: list[str], *, max_chars: int, max_chunks: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    for unit in units:
        for piece in _split_oversized_unit(unit, max_chars=max_chars):
            if not piece:
                continue
            if not current:
                current = piece
                continue

            candidate = f"{current} {piece}"
            if len(candidate) <= max_chars:
                current = candidate
                continue

            chunks.append(current)
            if len(chunks) >= max_chunks:
                return chunks
            current = piece

    if current and len(chunks) < max_chunks:
        chunks.append(current)
    return chunks


def _split_oversized_unit(unit: str, *, max_chars: int) -> list[str]:
    if len(unit) <= max_chars:
        return [unit]

    clause_parts = [part.strip() for part in CLAUSE_BOUNDARY_RE.split(unit) if part and part.strip()]
    if len(clause_parts) > 1:
        packed = _pack_units(clause_parts, max_chars=max_chars, max_chunks=64)
        if all(len(part) <= max_chars for part in packed):
            return packed

    return _split_by_words_or_hard_cut(unit, max_chars=max_chars)


def _split_by_words_or_hard_cut(text: str, *, max_chars: int) -> list[str]:
    words = text.split()
    if len(words) <= 1:
        return [text[i : i + max_chars].strip() for i in range(0, len(text), max_chars) if text[i : i + max_chars].strip()]

    chunks: list[str] = []
    current = ""
    for word in words:
        if not current:
            if len(word) <= max_chars:
                current = word
                continue
            chunks.extend(
                segment.strip()
                for segment in [word[i : i + max_chars] for i in range(0, len(word), max_chars)]
                if segment.strip()
            )
            continue

        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            if len(word) <= max_chars:
                current = word
            else:
                chunks.extend(
                    segment.strip()
                    for segment in [word[i : i + max_chars] for i in range(0, len(word), max_chars)]
                    if segment.strip()
                )
                current = ""

    if current:
        chunks.append(current)
    return chunks
