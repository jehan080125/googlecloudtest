"""Addressee routing and free-dialogue orchestration for the courtroom."""

from dataclasses import dataclass, field

from backend.schemas.episode import StageType, TrialStage

PROSECUTOR_KEYWORDS = ("검사", "검찰", "기소", "공소", "검사님", "검찰측")
WITNESS_KEYWORDS = (
    "증인",
    "증인님",
    "선생",
    "당신",
    "너",
    "너는",
    "너가",
    "야마노",
    "이소은",
    "소은",
    "임민수",
    "민수",
    "양진혁",
)
JUDGE_TRIGGER_KEYWORDS = ("판사", "재판장", "판결", "허락", "기록해", "판사님")
CROSS_EXAM_TURN_DEPTH = 3

DECISIVE_POINT_KEYWORDS = (
    "모순",
    "contradiction",
    "틀렸",
    "거짓",
    "부검",
    "증거와",
    "맞지 않",
    "불가능",
    "거짓말",
    "모순됩니다",
    "모순입니다",
    "말이 안",
    "사실과 다",
    "증언이 틀",
    "알리바",
    "시간대",
)


@dataclass
class OrchestrationPlan:
    """Who responds to a player turn and in what order."""

    responders: list[str] = field(default_factory=list)
    primary_addressee: str = "wit_001"
    trigger_judge_evaluation: bool = False
    judge_trigger: str = "none"
    reason: str = ""


def route_addressee(text: str, stage: TrialStage) -> str:
    """Return character id (pros_001 or active witness) from player message."""
    pros_score = sum(1 for keyword in PROSECUTOR_KEYWORDS if keyword in text)
    wit_score = sum(1 for keyword in WITNESS_KEYWORDS if keyword in text)

    if pros_score > wit_score:
        return "pros_001"
    if wit_score > pros_score:
        return stage.active_witness_id or "wit_001"

    if stage.stage_type == StageType.VS_PROSECUTOR:
        return "pros_001"
    return stage.active_witness_id or "wit_001"


def addressee_to_responder(addressee: str) -> str:
    if addressee == "pros_001":
        return "prosecutor"
    return "witness"


def detect_decisive_point(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in text or keyword in lowered for keyword in DECISIVE_POINT_KEYWORDS)


def should_trigger_passive_judge(text: str, exchange_count: int, threshold: int = 3) -> bool:
    if any(keyword in text for keyword in JUDGE_TRIGGER_KEYWORDS):
        return True
    return exchange_count > 0 and exchange_count % threshold == 0


def _cross_exam_witness_chain(*, include_followup: bool = True, turn_depth: int = CROSS_EXAM_TURN_DEPTH) -> list[str]:
    """Witness question chain: witness answers, prosecutor defends, optional witness follow-up."""
    responders = ["witness", "prosecutor"]
    if include_followup and turn_depth >= 3:
        responders.append("witness_followup")
    return responders


def _append_judge_evaluation(responders: list[str]) -> list[str]:
    """Run full judge evaluation after a cross-exam rebuttal chain."""
    if "judge" in responders:
        return responders
    return [*responders, "judge"]


def orchestrate_player_turn(
    text: str,
    stage: TrialStage,
    exchange_count: int,
    mode: str = "question",
    *,
    threshold: int = 3,
    stage_phase: str = "testimony",
    turn_depth: int = CROSS_EXAM_TURN_DEPTH,
) -> OrchestrationPlan:
    """Decide responder order for a player-initiated free dialogue turn."""
    addressee = route_addressee(text, stage)

    if mode == "objection":
        return OrchestrationPlan(
            responders=["judge"],
            primary_addressee=addressee,
            trigger_judge_evaluation=True,
            judge_trigger="objection",
            reason="이의 제기 — 판사 평가 필수",
        )

    judge_direct = any(keyword in text for keyword in JUDGE_TRIGGER_KEYWORDS)
    decisive = detect_decisive_point(text)
    passive = should_trigger_passive_judge(text, exchange_count, threshold)

    if judge_direct:
        return OrchestrationPlan(
            responders=["judge"],
            primary_addressee=addressee,
            trigger_judge_evaluation=True,
            judge_trigger="direct_address",
            reason="변호인이 판사에게 직접 말함",
        )

    primary = addressee_to_responder(addressee)
    cross_exam_chain = (
        _cross_exam_witness_chain(turn_depth=turn_depth)
        if stage_phase == "cross_exam_free" and primary == "witness"
        else None
    )
    cross_exam_prosecutor_only = (
        stage_phase == "cross_exam_free" and primary == "prosecutor"
    )

    if decisive:
        if cross_exam_chain:
            return OrchestrationPlan(
                responders=_append_judge_evaluation(cross_exam_chain),
                primary_addressee=addressee,
                trigger_judge_evaluation=True,
                judge_trigger="decisive",
                reason="증언·증거와의 결정적 모순/지적 — 반박 후 판사 턴 평가",
            )
        return OrchestrationPlan(
            responders=[primary, "judge"],
            primary_addressee=addressee,
            trigger_judge_evaluation=True,
            judge_trigger="decisive",
            reason="증언·증거와의 결정적 모순/지적 감지",
        )

    if passive:
        if cross_exam_chain:
            return OrchestrationPlan(
                responders=_append_judge_evaluation(cross_exam_chain),
                primary_addressee=addressee,
                trigger_judge_evaluation=True,
                judge_trigger="passive",
                reason=f"자유 질의 {exchange_count}회차 — 반박 후 판사 턴 평가",
            )
        responders = [primary, "judge"]
        if cross_exam_prosecutor_only:
            responders = _append_judge_evaluation(["prosecutor"])
        return OrchestrationPlan(
            responders=responders,
            primary_addressee=addressee,
            trigger_judge_evaluation=True,
            judge_trigger="passive",
            reason=f"자유 질의 {exchange_count}회차 — 판사 개입",
        )

    if stage_phase == "cross_exam_free":
        if cross_exam_chain:
            return OrchestrationPlan(
                responders=cross_exam_chain,
                primary_addressee=addressee,
                trigger_judge_evaluation=False,
                judge_trigger="none",
                reason="자유 반론 — 증인·검사·증인 연속 티키타카",
            )
        return OrchestrationPlan(
            responders=["prosecutor"],
            primary_addressee=addressee,
            trigger_judge_evaluation=False,
            judge_trigger="none",
            reason="자유 반론 — 검사 직접 응답",
        )

    return OrchestrationPlan(
        responders=[primary],
        primary_addressee=addressee,
        trigger_judge_evaluation=False,
        judge_trigger="none",
        reason="일반 질의 — 1차 응답자만",
    )
