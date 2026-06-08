import json
import os
from typing import Any, Optional

from pydantic import BaseModel, Field

from backend.ai_services.openai_structured import parse_openai_structured
from backend.config import LLM_PROVIDER, OPENAI_JUDGE_MODEL, get_openai_api_key
from backend.core.dialogue_chunker import split_dialogue_chunks
from backend.core.korean_name_sanitizer import CHARACTER_NAME_PROMPT_RULE, sanitize_character_names
from backend.logging_config import get_logger
from backend.schemas.court import ActorLine, ActorResponse
from backend.schemas.trial import (
    AnswerVerdict,
    ContradictionSeverity,
    DefenseArgumentEvaluation,
    RelevanceLevel,
    ScoringResult,
    TurnContradictionEvaluation,
)

logger = get_logger(__name__)
MAX_DIALOGUE_CHUNK_LEN = 160
MAX_DIALOGUE_CHUNKS_PER_LINE = 4
WITNESS_VOICE_MARKERS = (
    "저는 ",
    "제가 ",
    "제 손",
    "조금만 묻",
    "튀긴 했지만",
    "치사량보다",
)

JUDGE_NEUTRAL_ARBITER_BASE = (
    "당신은 중립적인 재판장입니다. 검찰·변호인 어느 쪽도 이기게 두지 말고, "
    "오직 제출된 주장·증거·증언의 논리적 일관성만 검증하세요. "
    "검사나 증인이 회피·억지·자기모순·비논리적 주장을 하면 판사로서 질서를 유지하고 "
    "명확한 답변을 요구하세요. "
    "검찰 주장이 약하거나 증언에 모순이 있어도 '증인 발언이 일관적이니 괜찮다'처럼 "
    "검찰 편을 들어주지 마세요. "
    "변호인의 타당한 이의·모순 지적을 논리적 근거 없이 기각하지 마세요. "
    "절대 증인 1인칭('저는', '제가', '제 손')으로 말하지 마세요. "
    "증인의 대사를 대신 읊거나 증인 시점으로 변호하지 말고, "
    "항상 판사 3인칭 어조로만 평가하세요. "
    "새 사실·새 증거·점수·생명·클리어 여부를 만들지 마세요."
)


class TurnContradictionLLMResult(BaseModel):
    intervention_needed: bool = False
    severity: ContradictionSeverity = ContradictionSeverity.NONE
    verdict: AnswerVerdict = AnswerVerdict.FAIL
    reason: str = ""
    life_loss: int = Field(default=0, ge=0)
    persuasion_delta: int = Field(default=0, ge=0)
    lines: list[ActorLine] = Field(default_factory=list)


class JudgeActorLLM:
    def __init__(self, api_key: Optional[str] = None):
        self.openai_api_key = get_openai_api_key("judge", api_key)
        self.openai_model = OPENAI_JUDGE_MODEL
        disable_config_llm = api_key is None and bool(os.getenv("PYTEST_CURRENT_TEST"))
        self._use_openai = (
            not disable_config_llm
            and bool(self.openai_api_key)
            and LLM_PROVIDER in ("auto", "openai")
        )

    async def evaluate_free_dialogue(
        self,
        *,
        stage_type: str,
        stage_id: str | None = None,
        event_type: str,
        evaluation: DefenseArgumentEvaluation | dict[str, Any],
        user_text: str,
        current_statement: dict[str, Any] | None,
        selected_evidence_ids: list[str],
        trigger: str,
        remaining_life: int = 0,
        witness_mental: int | None = None,
        judge_persuasion: int | None = None,
        verifier_feedback: str | None = None,
    ) -> ActorResponse:
        """Judge feedback for free dialogue / objection — must explain WHY good or bad."""
        parsed = self._coerce_evaluation(evaluation)
        if parsed is None:
            parsed = DefenseArgumentEvaluation(
                relevance=RelevanceLevel.PARTIALLY_RELEVANT,
                core_match_score=0.3,
                logic_score=0.3,
                evidence_usage_score=0.0,
                verdict=AnswerVerdict.FAIL,
                reason="평가 정보가 없습니다.",
            )

        payload = {
            "stage_type": stage_type,
            "event_type": event_type,
            "stage_id": stage_id or "",
            "trigger": trigger,
            "evaluation": parsed.model_dump(),
            "user_text": user_text,
            "current_statement": current_statement or {},
            "selected_evidence_ids": selected_evidence_ids,
            "remaining_life": remaining_life,
            "witness_mental": witness_mental,
            "judge_persuasion": judge_persuasion,
            "verifier_feedback": verifier_feedback or "",
        }
        system = (
            f"{JUDGE_NEUTRAL_ARBITER_BASE} "
            "변호인의 자유 발언 또는 이의에 대해 evaluation.reason과 verdict를 근거로 "
            "논리적 타당성만 설명하세요. "
            "verdict가 success면 현재 증언·증거 사이의 모순/결함을 유효히 짚었다고, "
            "fail이면 아직 논리적 연결이 부족하다고 이유를 밝히세요. "
            "검찰 반박이 단순 주장만으로 변호인 지적을 무효화했다고 말하지 마세요. "
            f"{self._trial_specific_guardrail(stage_id)} "
            f"{self._verification_feedback_instruction(verifier_feedback)} "
            f"한국어 1~2줄, 각 줄 90자 이하. JSON lines만 반환하세요. {CHARACTER_NAME_PROMPT_RULE}"
        )

        if self._use_openai:
            try:
                return self._sanitize(
                    await parse_openai_structured(
                        api_key=self.openai_api_key,
                        model=self.openai_model,
                        system=system,
                        user=json.dumps(payload, ensure_ascii=False),
                        response_model=ActorResponse,
                        temperature=0.35,
                    ),
                    event_type,
                    parsed,
                    None,
                    selected_evidence_ids,
                )
            except Exception as e:
                logger.warning("JudgeActor evaluate_free_dialogue OpenAI failed: %s", e)

        return self._mock_free_dialogue_comment(
            event_type, parsed, trigger, selected_evidence_ids, user_text
        )

    @staticmethod
    def _trial_specific_guardrail(stage_id: str | None) -> str:
        if stage_id == "stage_epitaph_club":
            return (
                "1차 재판 핵심은 배틀 순서대로 판단하세요: "
                "먼저 증언 #1과 VX 용량(20mg/50mg), 다음 증언 #1·#2 순서 모순, 마지막으로 증언 #4와 피부 치사량(10mg)입니다."
            )
        if stage_id != "stage_epitaph_car":
            return ""
        return (
            "2차 재판 3배틀 순서: "
            "①CCTV 우회전·서버 로그 좌회전 모순 ②LiDAR 소견 희박·살의면 우회전 코드 "
            "③카카오=소호 보복 의심, 이소은 살해 동기 미증명. "
            "판사는 무죄추정 원칙을 지키며 동기만으로 유죄 단정하지 마세요."
        )

    @staticmethod
    def _verification_feedback_instruction(verifier_feedback: str | None) -> str:
        if not verifier_feedback:
            return ""
        return (
            "이전 초안이 논리 검증에서 반려되었습니다. 다음 수정 지시를 반영하세요: "
            f"{verifier_feedback}"
        )

    async def evaluate_turn_contradiction(
        self,
        *,
        stage_type: str,
        stage_phase: str,
        mode: str,
        user_text: str,
        turn_batch_lines: list[dict[str, str]],
        current_statement: dict[str, Any] | None,
        remaining_life: int = 0,
        witness_mental: int | None = None,
        judge_persuasion: int | None = None,
    ) -> tuple[TurnContradictionEvaluation, ActorResponse]:
        """Evaluate whether rebuttals severely undermined the defense argument this turn."""
        payload = {
            "stage_type": stage_type,
            "stage_phase": stage_phase,
            "mode": mode,
            "user_text": user_text,
            "turn_batch_lines": turn_batch_lines,
            "current_statement": current_statement or {},
            "remaining_life": remaining_life,
            "witness_mental": witness_mental,
            "judge_persuasion": judge_persuasion,
        }
        system = (
            f"{JUDGE_NEUTRAL_ARBITER_BASE}\n"
            "이번 턴 전체(변호인 발언 + 증인/검사 반박)의 논리적 일관성을 평가하세요.\n"
            "판단 기준:\n"
            "- severity=none: 변호인 논리가 유지되거나, 증인/검사 반박이 논리적으로 설득력 없음. "
            "증인·검사가 회피·억지·자기모순이면 변호인에게 불이익 주지 말고 "
            "증인/검사에게 명확히 답하라고 요구(intervention_needed=false).\n"
            "- severity=minor: 변호인 주장 일부가 흔들렸으나 재반론 여지 있음 "
            "→ intervention_needed=false, life_loss=0\n"
            "- severity=severe: 변호인 발언이 터무니없거나 비논리적이거나, "
            "반박이 논리적으로 변호인 지적을 명확히 무효화함 "
            "→ intervention_needed=true, life_loss=1(필수), persuasion_delta=0\n"
            "- intervention_needed=true이면 반드시 life_loss=1을 설정하세요.\n"
            "- 검사의 단순 반박('성립하지 않습니다')만으로 severe 판정하지 마세요.\n"
            "- '증인 발언이 일관적이니 괜찮다' 등 검찰 편 든 표현 금지.\n"
            f"새 사실·증거를 만들지 마세요. 한국어 1~2줄, 각 줄 90자 이하. JSON만 반환. {CHARACTER_NAME_PROMPT_RULE}"
        )

        if self._use_openai:
            try:
                result = await parse_openai_structured(
                    api_key=self.openai_api_key,
                    model=self.openai_model,
                    system=system,
                    user=json.dumps(payload, ensure_ascii=False),
                    response_model=TurnContradictionLLMResult,
                    temperature=0.35,
                )
                return self._sanitize_turn_result(result, remaining_life=remaining_life)
            except Exception as e:
                logger.warning("JudgeActor evaluate_turn_contradiction OpenAI failed: %s", e)

        return self._mock_turn_contradiction(
            user_text=user_text,
            turn_batch_lines=turn_batch_lines,
            mode=mode,
            remaining_life=remaining_life,
        )

    def _sanitize_turn_result(
        self,
        result: TurnContradictionLLMResult,
        *,
        remaining_life: int = 0,
    ) -> tuple[TurnContradictionEvaluation, ActorResponse]:
        intervention_needed = result.intervention_needed or result.severity == ContradictionSeverity.SEVERE
        severity = (
            ContradictionSeverity.SEVERE
            if intervention_needed and result.severity != ContradictionSeverity.SEVERE
            else result.severity
        )
        life_loss = 0
        if intervention_needed and remaining_life > 0:
            life_loss = max(result.life_loss, 1)

        evaluation = TurnContradictionEvaluation(
            intervention_needed=intervention_needed,
            severity=severity,
            verdict=result.verdict,
            reason=(result.reason or "").strip(),
            life_loss=life_loss,
            persuasion_delta=result.persuasion_delta,
        )
        if not intervention_needed:
            return evaluation, ActorResponse(lines=[])

        lines = []
        for line in result.lines[:2]:
            chunks = split_dialogue_chunks(
                sanitize_character_names((line.dialogue or "").strip()),
                max_chars=MAX_DIALOGUE_CHUNK_LEN,
                max_chunks=MAX_DIALOGUE_CHUNKS_PER_LINE,
            )
            tag = (
                line.animation_tag
                if line.animation_tag not in ("idle", "normal")
                else self._animation_for_event(
                    "passive_intervention" if intervention_needed else "argument_fail",
                    evaluation,
                    intervention=intervention_needed,
                )
            )
            for dialogue in chunks:
                if not dialogue or self._sounds_like_witness_voice(dialogue):
                    continue
                lines.append(
                    ActorLine(
                        speaker="judge_001",
                        dialogue=dialogue,
                        animation_tag=tag,
                    )
                )
        if not lines:
            lines = self._mock_turn_judge_lines(evaluation, result.reason).lines
        return evaluation, ActorResponse(lines=lines)

    def _mock_turn_contradiction(
        self,
        *,
        user_text: str,
        turn_batch_lines: list[dict[str, str]],
        mode: str,
        remaining_life: int,
    ) -> tuple[TurnContradictionEvaluation, ActorResponse]:
        witness_rebuttal = " ".join(
            entry.get("text", "")
            for entry in turn_batch_lines
            if "wit" in entry.get("speaker", "")
        )
        prosecutor_rebuttal = " ".join(
            entry.get("text", "")
            for entry in turn_batch_lines
            if entry.get("speaker", "").startswith("pros")
        )
        combined_rebuttal = f"{witness_rebuttal} {prosecutor_rebuttal}".strip()
        player_substantive = any(
            keyword in user_text
            for keyword in (
                "모순",
                "증거",
                "부검",
                "맞지 않",
                "불가능",
                "틀렸",
                "거짓",
                "CCTV",
                "로그",
                "용량",
                "치사량",
            )
        )
        player_nonsense = any(
            keyword in user_text
            for keyword in (
                "외계",
                "마법",
                "우주",
                "용이",
                "터무니",
                "말도 안",
                "엉터리",
                "완전히 틀",
                "근거 없",
            )
        )
        witness_evasive = any(
            phrase in witness_rebuttal
            for phrase in (
                "기억",
                "모르",
                "확실하지",
                "……",
                "회피",
                "답하기 어렵",
            )
        )
        logical_rebuttal = any(
            phrase in prosecutor_rebuttal
            for phrase in (
                "수치",
                "용량",
                "mg",
                "CCTV",
                "서버",
                "소견서",
                "치사량",
                "부검",
                "사망 시각",
            )
        )
        empty_rebuttal = any(
            phrase in prosecutor_rebuttal
            for phrase in (
                "성립하지 않",
                "받아들일 수 없",
                "핵심 쟁점",
            )
        ) and not logical_rebuttal

        if player_nonsense and turn_batch_lines:
            evaluation = TurnContradictionEvaluation(
                intervention_needed=True,
                severity=ContradictionSeverity.SEVERE,
                verdict=AnswerVerdict.FAIL,
                reason="변호인의 발언은 사건과 무관하거나 논리적 근거가 부족합니다.",
                life_loss=1 if remaining_life > 0 else 0,
                persuasion_delta=0,
            )
            return evaluation, self._mock_turn_judge_lines(evaluation, evaluation.reason)

        if witness_evasive and player_substantive:
            evaluation = TurnContradictionEvaluation(
                intervention_needed=False,
                severity=ContradictionSeverity.NONE,
                verdict=AnswerVerdict.PARTIAL_SUCCESS,
                reason="증인/검사의 답변이 회피적입니다. 명확한 답변을 요구합니다.",
            )
            return evaluation, self.build_turn_sustain_response(evaluation, evaluation.reason)

        if empty_rebuttal and player_substantive:
            evaluation = TurnContradictionEvaluation(
                intervention_needed=False,
                severity=ContradictionSeverity.NONE,
                verdict=AnswerVerdict.PARTIAL_SUCCESS,
                reason="검사의 반박만으로는 변호인 지적이 무효화되지 않습니다.",
            )
            return evaluation, self.build_turn_sustain_response(evaluation, evaluation.reason)

        if logical_rebuttal and not player_substantive and turn_batch_lines:
            evaluation = TurnContradictionEvaluation(
                intervention_needed=True,
                severity=ContradictionSeverity.SEVERE,
                verdict=AnswerVerdict.FAIL,
                reason="변호인의 지적은 반박에 의해 논리적 근거가 충분히 뒷받침되지 못했습니다.",
                life_loss=1 if remaining_life > 0 else 0,
                persuasion_delta=0,
            )
            return evaluation, self._mock_turn_judge_lines(evaluation, evaluation.reason)

        evaluation = TurnContradictionEvaluation(
            intervention_needed=False,
            severity=ContradictionSeverity.NONE,
            verdict=AnswerVerdict.FAIL,
            reason="이번 턴은 판사 개입이 필요하지 않습니다.",
        )
        return evaluation, ActorResponse(lines=[])

    def _mock_turn_judge_lines(
        self, evaluation: TurnContradictionEvaluation, reason: str
    ) -> ActorResponse:
        text = (
            "변호인, 방금 주장은 제출된 증거·논리로 충분히 뒷받침되지 못했습니다. "
            "쟁점과 연결된 근거를 제시하십시오."
        )
        if reason and reason not in text:
            text = f"{text} {reason}"
        if evaluation.life_loss > 0:
            text = f"{text} (생명 -{evaluation.life_loss})"
        return ActorResponse(
            lines=[ActorLine(speaker="judge_001", dialogue=text[:200], animation_tag="serious")]
        )

    def build_turn_sustain_response(
        self, evaluation: TurnContradictionEvaluation, reason: str
    ) -> ActorResponse:
        if evaluation.verdict == AnswerVerdict.SUCCESS:
            text = "변호인의 지적은 타당합니다. 증인, 회피하지 말고 명확히 답하십시오."
            tag = "success"
        else:
            text = (
                "검사의 반박만으로는 변호인 지적이 무효화되지 않습니다. "
                "증인, 질문에 분명히 답하십시오."
            )
            tag = "think"
        if reason and reason not in text:
            text = f"{text} {reason}"
        return ActorResponse(
            lines=[ActorLine(speaker="judge_001", dialogue=text[:200], animation_tag=tag)]
        )

    async def generate_stage_comment(
        self,
        *,
        stage_type: str,
        event_type: str,
        evaluation: DefenseArgumentEvaluation | dict[str, Any] | None,
        stage_result: dict[str, Any] | None,
        remaining_life: int,
        witness_mental: int | None,
        judge_persuasion: int | None,
        current_statement: dict[str, Any] | None,
        user_answer: str,
        selected_evidence_ids: list[str],
    ) -> ActorResponse:
        parsed_evaluation = self._coerce_evaluation(evaluation)
        payload = {
            "stage_type": stage_type,
            "event_type": event_type,
            "evaluation": parsed_evaluation.model_dump() if parsed_evaluation else evaluation,
            "stage_result": stage_result or {},
            "remaining_life": remaining_life,
            "witness_mental": witness_mental,
            "judge_persuasion": judge_persuasion,
            "current_statement": current_statement or {},
            "user_answer": user_answer,
            "selected_evidence_ids": selected_evidence_ids,
        }
        system = (
            f"{JUDGE_NEUTRAL_ARBITER_BASE} "
            "변호인은 현재 증인 발언을 상대로 논박하고 있습니다. "
            "evaluation과 stage_result만 근거로 논리적 결함·증거 모순 지적 여부만 판정하세요. "
            "성공이면 '증언의 모순/논리적 결함이 지적되었다'는 취지로, 실패면 "
            "'논리적 연결이 아직 부족하다'는 취지로 말하세요. "
            "검찰 주장을 옹호하거나 변호인 이의를 근거 없이 기각하지 마세요. "
            f"한국어 1~2줄, 각 줄 90자 이하. JSON lines만 반환하세요. {CHARACTER_NAME_PROMPT_RULE}"
        )

        if self._use_openai:
            try:
                return self._sanitize(
                    await parse_openai_structured(
                        api_key=self.openai_api_key,
                        model=self.openai_model,
                        system=system,
                        user=json.dumps(payload, ensure_ascii=False),
                        response_model=ActorResponse,
                        temperature=0.35,
                    ),
                    event_type,
                    parsed_evaluation,
                    stage_result,
                    selected_evidence_ids,
                )
            except Exception as e:
                logger.warning("JudgeActor OpenAI failed: %s", e)

        return self._mock_stage_comment(
            event_type, parsed_evaluation, stage_result, selected_evidence_ids
        )

    def final_verdict_lines(
        self,
        verdict: dict,
        weakened_claim_ids: list[str],
    ) -> ActorResponse:
        grade = verdict["grade"]
        total_score = verdict["total_score"]
        max_possible_score = verdict["max_possible_score"]
        score_ratio = verdict["score_ratio"]

        if grade in ("S", "A"):
            dialogue = "검찰과 증인의 핵심 논리는 더 이상 피고인의 유죄를 뒷받침하지 못합니다. 무죄입니다."
        elif grade == "B":
            dialogue = "변호인은 합리적 의심을 만드는 데 성공했습니다. 이 법정은 유죄를 단정할 수 없습니다."
        else:
            dialogue = "변호인의 논박은 충분하지 않았습니다. 증언의 결함을 더 명확히 밝혔어야 합니다."

        return ActorResponse(
            lines=[
                ActorLine(
                    speaker="judge_001",
                    dialogue=(
                        f"{dialogue} "
                        f"(총점 {total_score}/{max_possible_score}, 달성률 {score_ratio:.0%}, 등급 {grade})"
                    ),
                    animation_tag="idle",
                )
            ]
        )

    def round_comment(self, scoring: ScoringResult) -> ActorResponse:
        if scoring.passed:
            text = f"변호인의 논박을 인정합니다. 이번 증언의 결함은 지적되었습니다. 점수 {scoring.final_score}점."
        else:
            text = f"아직 부족합니다. 현재 증언의 모순을 더 분명히 지적하십시오. {scoring.feedback}"
        return ActorResponse(
            lines=[ActorLine(speaker="judge_001", dialogue=text, animation_tag="think")]
        )

    def _animation_for_event(
        self,
        event_type: str,
        evaluation: DefenseArgumentEvaluation | None,
        *,
        intervention: bool = False,
    ) -> str:
        if event_type in ("objection_sustained", "stage_cleared"):
            return "success"
        if event_type == "stage_failed":
            return "serious"
        if intervention or event_type in ("life_lost",):
            return "serious"
        if evaluation and evaluation.verdict == AnswerVerdict.SUCCESS:
            return "success"
        if evaluation and evaluation.verdict == AnswerVerdict.PARTIAL_SUCCESS:
            return "think"
        if event_type in ("objection_overruled", "objection_rejected", "passive_intervention"):
            return "think"
        return "think"

    def _coerce_evaluation(
        self, evaluation: DefenseArgumentEvaluation | dict[str, Any] | None
    ) -> DefenseArgumentEvaluation | None:
        if evaluation is None:
            return None
        if isinstance(evaluation, DefenseArgumentEvaluation):
            return evaluation
        return DefenseArgumentEvaluation.model_validate(evaluation)

    def _sanitize(
        self,
        response: ActorResponse,
        event_type: str,
        evaluation: DefenseArgumentEvaluation | None,
        stage_result: dict[str, Any] | None,
        selected_evidence_ids: list[str],
    ) -> ActorResponse:
        lines = []
        for line in response.lines[:2]:
            chunks = split_dialogue_chunks(
                sanitize_character_names((line.dialogue or "").strip()),
                max_chars=MAX_DIALOGUE_CHUNK_LEN,
                max_chunks=MAX_DIALOGUE_CHUNKS_PER_LINE,
            )
            tag = (
                line.animation_tag
                if line.animation_tag not in ("idle", "normal")
                else self._animation_for_event(event_type, evaluation)
            )
            for dialogue in chunks:
                if not dialogue:
                    continue
                if self._sounds_like_witness_voice(dialogue):
                    continue
                lines.append(
                    ActorLine(
                        speaker="judge_001",
                        dialogue=dialogue,
                        animation_tag=tag,
                    )
                )
        if lines:
            return ActorResponse(lines=lines)
        return self._mock_stage_comment(event_type, evaluation, stage_result, selected_evidence_ids)

    @staticmethod
    def _sounds_like_witness_voice(dialogue: str) -> bool:
        text = (dialogue or "").strip()
        if not text:
            return False
        return any(marker in text for marker in WITNESS_VOICE_MARKERS)

    def _mock_free_dialogue_comment(
        self,
        event_type: str,
        evaluation: DefenseArgumentEvaluation,
        trigger: str,
        selected_evidence_ids: list[str],
        user_text: str,
    ) -> ActorResponse:
        verdict = evaluation.verdict
        reason = (evaluation.reason or "").strip()
        base = self._mock_stage_comment(event_type, evaluation, None, selected_evidence_ids)
        dialogue = base.lines[0].dialogue if base.lines else ""

        if reason and reason not in dialogue:
            dialogue = f"{dialogue} ({reason})"
        elif trigger == "decisive" and verdict == AnswerVerdict.SUCCESS:
            dialogue = (
                "변호인, 방금 지적은 현재 증언과 증거 사이의 모순을 유효하게 드러냈습니다. "
                f"{reason or '논리적 결함이 명확합니다.'}"
            )
        elif trigger == "decisive" and verdict in (AnswerVerdict.FAIL, AnswerVerdict.IRRELEVANT):
            dialogue = (
                "그 지적은 방향은 맞을 수 있으나, 현재 증언의 결함을 아직 충분히 입증하지 못했습니다. "
                f"{reason or '증거와의 연결을 더 분명히 하십시오.'}"
            )
        elif trigger == "objection" and not selected_evidence_ids:
            dialogue = (
                "이의, 기각합니다. 증거 없이는 증언의 모순을 법정에서 인정하기 어렵습니다. "
                f"{reason or ''}"
            ).strip()

        return ActorResponse(
            lines=[
                ActorLine(
                    speaker="judge_001",
                    dialogue=dialogue[:200],
                    animation_tag=base.lines[0].animation_tag if base.lines else "think",
                )
            ]
        )

    def _mock_stage_comment(
        self,
        event_type: str,
        evaluation: DefenseArgumentEvaluation | None,
        stage_result: dict[str, Any] | None,
        selected_evidence_ids: list[str],
    ) -> ActorResponse:
        stage_result = stage_result or {}
        verdict = evaluation.verdict if evaluation else None

        if event_type == "objection_sustained":
            text = "이의, 인정합니다. 변호인의 지적은 현재 증언의 모순을 유효하게 드러냅니다."
            tag = "success"
        elif event_type == "objection_partial":
            text = "이의는 일부 인정합니다. 다만 그 지적이 증언 전체를 무너뜨리기엔 아직 부족합니다."
            tag = "think"
        elif event_type in ("objection_overruled", "objection_rejected"):
            text = (
                "이의, 기각합니다. 제시된 증거와 주장이 현재 증언의 모순을 "
                "논리적으로 입증하지 못했습니다."
            )
            tag = "think"
        elif event_type == "passive_intervention":
            text = "법정의 질서를 유지하겠습니다. 변호인, 질문은 현재 쟁점에 맞추십시오."
            tag = "think"
        elif event_type == "stage_cleared":
            text = "증인의 증언은 핵심 모순을 모두 지적당했습니다. 이 증언은 더 이상 믿기 어렵습니다."
            tag = "success"
        elif event_type == "stage_failed":
            text = "변호인은 현재 증언의 논리적 결함을 충분히 짚지 못했습니다. 이 신문은 실패로 보겠습니다."
            tag = "serious"
        elif event_type == "life_lost":
            text = "그 논박은 현재 증언의 약점을 찌르지 못했습니다. 증거와 증언의 충돌을 더 정확히 말하십시오."
            tag = "serious"
        elif not selected_evidence_ids:
            text = "변호인, 추측만으로는 증언을 흔들 수 없습니다. 모순을 뒷받침할 증거를 제시하십시오."
            tag = "think"
        elif verdict == AnswerVerdict.SUCCESS:
            text = "변호인의 지적은 타당합니다. 제시한 증거가 현재 증언의 논리적 결함을 드러냈습니다."
            tag = "success"
        elif verdict == AnswerVerdict.PARTIAL_SUCCESS:
            text = "방향은 맞습니다. 다만 그 증거가 현재 증언과 어떻게 충돌하는지 더 명확히 설명하십시오."
            tag = "think"
        else:
            text = "그 주장은 현재 증언의 결함과 충분히 연결되지 않습니다. 증언 안의 모순을 겨냥하십시오."
            tag = "think"

        if stage_result.get("stage_score"):
            text = f"{text} 스테이지 점수는 {stage_result['stage_score']}점입니다."

        return ActorResponse(lines=[ActorLine(speaker="judge_001", dialogue=text, animation_tag=tag)])
