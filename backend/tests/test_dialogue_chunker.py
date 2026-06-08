from backend.core.dialogue_chunker import split_dialogue_chunks


def test_split_dialogue_chunks_prefers_sentence_boundaries():
    text = (
        "첫 문장은 짧습니다. "
        "두 번째 문장은 조금 더 길어서 말풍선이 나뉘는지 확인하려고 일부러 길게 작성했습니다. "
        "세 번째 문장입니다."
    )
    chunks = split_dialogue_chunks(text, max_chars=70)

    assert len(chunks) >= 2
    assert chunks[0].endswith(".")
    assert all(len(chunk) <= 70 for chunk in chunks)


def test_split_dialogue_chunks_keeps_words_intact_when_possible():
    text = " ".join(["word"] * 35)
    chunks = split_dialogue_chunks(text, max_chars=40)

    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)
    assert all(not chunk.startswith(" ") and not chunk.endswith(" ") for chunk in chunks)


def test_split_dialogue_chunks_hard_cuts_single_long_token():
    text = "가" * 390
    chunks = split_dialogue_chunks(text, max_chars=120)

    assert len(chunks) == 4
    assert all(len(chunk) <= 120 for chunk in chunks)
