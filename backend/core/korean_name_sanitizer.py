"""Post-process Korean dialogue to fix incorrectly spaced character names."""

import re

_NAME_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"이\s+소은"), "이소은"),
    (re.compile(r"양\s+진혁"), "양진혁"),
    (re.compile(r"임\s+민수"), "임민수"),
    (re.compile(r"앤\s+서니", re.IGNORECASE), "앤서니"),
    (re.compile(r"Anthony", re.IGNORECASE), "앤서니"),
    (re.compile(r"소호\s*차량을"), "연행 호송차를"),
    (re.compile(r"소호의\s*차량"), "연행 호송차"),
    (re.compile(r"소호\s*차량"), "연행 호송차"),
)

CHARACTER_NAME_PROMPT_RULE = (
    "등장인물 이름은 반드시 붙여 쓸 것: '이소은'(증인), '양진혁', '임민수', '앤서니'(2·3차 피고인). "
    "'이 소은', '양 진혁', '임 민수', '앤 서니' 등 띄어쓰기 금지. 영문 Anthony는 '앤서니'로 쓸 것. "
    "2차 재판(차량 해킹)에서 해킹 대상은 '연행 호송차' 또는 '경찰 호송차'이며, "
    "'소호 차량'·'소호차량'은 바텐더 소호(1차 피고인)와 혼동한 잘못된 표현이므로 절대 쓰지 말 것."
)


def sanitize_character_names(text: str) -> str:
    if not text:
        return text
    result = text
    for pattern, replacement in _NAME_FIXES:
        result = pattern.sub(replacement, result)
    return result
