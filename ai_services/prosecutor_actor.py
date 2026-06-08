import json
import os
from typing import Any, Optional

from backend.ai_services.openai_structured import parse_openai_structured
from backend.config import LLM_PROVIDER, OPENAI_PROSECUTOR_MODEL, get_openai_api_key
from backend.core.dialogue_chunker import split_dialogue_chunks
from backend.core.korean_name_sanitizer import CHARACTER_NAME_PROMPT_RULE, sanitize_character_names
from backend.logging_config import get_logger
from backend.schemas.court import ActorLine, ActorResponse
from backend.schemas.episode import EpisodeData, ProsecutionClaim
from backend.schemas.trial import (
    AnswerEvaluationResult,
    DefenseArgumentEvaluation,
    ProsecutorPlan,
    ProsecutorPlanMode,
)

logger = get_logger(__name__)

MAX_DIALOGUE_CHUNK_LEN = 160
MAX_DIALOGUE_CHUNKS_PER_LINE = 4


PROSECUTOR_BASE = (
    "당신은 법정 게임의 검사입니다. 당신의 역할은 변호인(피고 측)의 주장을 반박하는 것입니다. "
    "증인의 증언은 검찰이 제출한 핵심 증거이며, 당신은 그 증언을 반드시 옹호·방어해야 합니다. "
    "절대 증인의 말을 의심하거나, 증언이 틀렸다고 하거나, 증인을 약하게 만드는 발언을 하지 마세요. "
    "증인 증언이 피고인의 유죄를 입증한다는 검찰 내러티브를 유지하세요. "
    "반박 대상은 오직 변호인의 질문·주장·논리뿐입니다. "
    "변호인이 찾아야 할 모순, 약점, 정답 증거, 반박 방법은 절대 말하지 마세요. "
    "새 사실이나 새 증거를 만들지 마세요. "
    "법정 드라마 톤으로 짧고 강렬한 대사를 쓰세요. "
    "animation_tag 기본값은 basic(또는 idle)이며, 감정이 고조된 순간에만 objection, angry, pressure를 쓰세요. "
    "JSON lines만 반환하세요. "
    f"{CHARACTER_NAME_PROMPT_RULE}"
)

PROSECUTOR_UNDERMINE_PHRASES = (
    "증인의 말이 틀",
    "증언이 틀",
    "증인이 거짓",
    "증인이 잘못",
    "증언이 잘못",
    "증인의 주장이 성립하지",
    "증인을 신뢰할 수 없",
    "증언을 믿을 수 없",
    "증인의 증언은 약",
    "증인의 말은 믿기 어렵",
)


class ProsecutorActorLLM:
    def __init__(self, api_key: Optional[str] = None):
        self.openai_api_key = get_openai_api_key("prosecutor", api_key)
        self.openai_model = OPENAI_PROSECUTOR_MODEL
        disable_config_llm = api_key is None and bool(os.getenv("PYTEST_CURRENT_TEST"))
        self._use_openai = (
            not disable_config_llm
            and bool(self.openai_api_key)
            and LLM_PROVIDER in ("auto", "openai")
        )

    async def generate(
        self,
        plan: ProsecutorPlan,
        claim: ProsecutionClaim,
        evidence_details: list[dict[str, Any]],
        witness_text: str,
        episode: EpisodeData,
        user_answer: Optional[str] = None,
        evaluation: Optional[AnswerEvaluationResult] = None,
    ) -> ActorResponse:
        payload = {
            "mode": plan.mode.value,
            "plan": plan.model_dump(),
            "claim": claim.model_dump(),
            "evidence_details": evidence_details,
            "witness_text": witness_text,
            "user_answer": user_answer,
            "evaluation": evaluation.model_dump() if evaluation else None,
            "forbidden_claims": (episode.prosecution_case.forbidden_claims if episode.prosecution_case else [])
            + episode.forbidden_claims,
        }
        system = (
            f"{PROSECUTOR_BASE} opening이면 증언과 증거가 검찰 주장에 어떻게 도움 되는지 말하세요. "
            "retreat이면 방금 논박으로 해당 주장 일부가 약해졌음을 인정하되, 아직 검찰이 물러서지 않는 태도를 보이세요."
        )
        return await self._generate_or_mock(payload, system, plan.mode, self._mock(plan, claim, evidence_details))

    async def respond_to_free_question(
        self,
        *,
        user_question: str,
        stage: Any,
        episode: EpisodeData,
        dialogue_context: dict[str, Any] | None = None,
        verifier_feedback: str | None = None,
    ) -> ActorResponse:
        dialogue_context = dialogue_context or {}
        character = dialogue_context.get("character") or {}
        response_mode = dialogue_context.get("response_mode", "counter_defense")
        max_lines = dialogue_context.get("max_response_lines", 2)
        payload = {
            "role": dialogue_context.get("role", "prosecutor"),
            "character": character,
            "user_question": user_question,
            "response_mode": response_mode,
            "turn_batch_lines": dialogue_context.get("turn_batch_lines") or [],
            "batch_position": dialogue_context.get("batch_position", 0),
            "stage_id": stage.stage_id,
            "stage_context": dialogue_context.get("stage_context") or {},
            "prosecution_context": getattr(stage, "prosecution_context", None) or {},
            "prosecution_case": dialogue_context.get("prosecution_case") or {},
            "case_summary": dialogue_context.get("case_summary") or {},
            "dialogue_history": dialogue_context.get("dialogue_history") or [],
            "character_knowledge": dialogue_context.get("character_knowledge") or {},
            "inventory_evidence": dialogue_context.get("inventory_evidence") or [],
            "exchange_count": dialogue_context.get("exchange_count", 0),
            "forbidden_claims": episode.forbidden_claims,
            "verifier_feedback": verifier_feedback or "",
        }
        if response_mode == "defend_witness":
            mode_instruction = (
                "방금 증인이 한 말(turn_batch_lines)을 직접 옹호하세요. "
                "변호인의 공격(user_question)을 한 줄 한 줄 반박하고, "
                "증인의 증언이 왜 신뢰할 만한지·피고인에게 왜 불리한지 강조하세요. "
                "증인의 말을 수정·축소·의심하지 마세요. "
                "예: '변호인의 주장은 성립하지 않습니다! 증인의 증언은...' "
                "같은 턴에서 이미 나온 대사에 이어지는 연속 대화처럼 쓰세요."
            )
        else:
            mode_instruction = (
                "변호인이 검사에게 직접 한 질문(user_question)에 답하세요. "
                "검찰의 유죄 논리를 방어하고 변호인의 주장을 반박하세요. "
                "증인 증언이 검찰 주장을 뒷받침한다는 점을 강조하세요."
            )
        system = (
            f"{PROSECUTOR_BASE} 당신은 검사 '{character.get('name', '검사')}'입니다. "
            f"{mode_instruction} "
            "character, prosecution_case, prosecution_context, case_summary, dialogue_history, "
            "turn_batch_lines, character_knowledge, inventory_evidence를 참고하세요. "
            "이전 대화 맥락(dialogue_history)과 같은 턴의 선행 대사(turn_batch_lines)를 반드시 이어가세요. "
            "character_knowledge.scope에 허용된 정보만 사용하세요. "
            f"{self._trial_specific_guardrail(dialogue_context)} "
            f"{self._verification_feedback_instruction(verifier_feedback)} "
            f"한국어 1~{max_lines}줄, 각 줄 100자 이하. JSON lines만 반환하세요."
        )
        return await self._generate_or_mock(
            payload,
            system,
            ProsecutorPlanMode.PRESSURE,
            self._mock_free_question(user_question, dialogue_context),
            max_lines=max_lines,
            response_mode=response_mode,
        )

    def _trial_specific_guardrail(self, dialogue_context: dict[str, Any]) -> str:
        stage_id = ((dialogue_context.get("stage_context") or {}).get("stage_id")) or ""
        if stage_id == "stage_epitaph_club":
            return (
                "현재는 1차 재판(클럽 VX 살인)입니다. 증인 이름은 반드시 '이소은'으로 붙여 쓸 것. "
                "'이 소은' 금지. 검찰은 먼저 '바텐더 독살' 프레임을 유지하고, "
                "변호인이 용량(20mg/50mg) 모순을 찌르면 심장질환 소견서 반박으로 방어하세요."
            )
        if stage_id != "stage_epitaph_car":
            return ""
        return (
            "현재는 2차 재판(차량 해킹)입니다. 피고인은 '앤서니'(YJ社 선임 엔지니어)입니다. "
            "해킹 대상 차량은 이소은 연행 '경찰 호송차'(또는 '연행 호송차')이며, "
            "바텐더 '소호'의 차량이 아닙니다. '소호 차량'·'소호차량' 표현은 절대 쓰지 마세요. "
            "소호는 1차 재판 피고인(바텐더) 이름일 뿐 2차 사건 차량과 무관합니다. "
            "임민수 증인의 축은 "
            "「앤서니 해킹·우회전 주장 → 라이다 글리치 변명 → 카카오 동기 강조」입니다. "
            "검사는 이 축을 강화해야 하며, 절대 스스로 깨지 마세요. "
            "핵심 증거 축: CCTV 실제 우회전 vs 서버 로그 좌회전 명령. "
            "임민수 증언은 우회전 해킹을 주장하나 서버 로그는 좌회전이다. "
            "로그 위조 가능성·카카오 동기는 변호인이 들이밀기 전 선제 고백하지 마세요."
        )

    @staticmethod
    def _verification_feedback_instruction(verifier_feedback: str | None) -> str:
        if not verifier_feedback:
            return ""
        return (
            "이전 초안이 논리 검증에서 반려되었습니다. 다음 수정 지시를 반드시 반영해 다시 작성하세요: "
            f"{verifier_feedback}"
        )

    def _mock_free_question(self, user_question: str, dialogue_context: dict[str, Any]) -> ActorResponse:
        response_mode = dialogue_context.get("response_mode", "counter_defense")
        turn_batch_lines = dialogue_context.get("turn_batch_lines") or []
        witness_line = next(
            (entry.get("text", "") for entry in reversed(turn_batch_lines) if entry.get("speaker", "").startswith("wit")),
            "",
        )
        stage_context = dialogue_context.get("stage_context") or {}
        prosecution_context = stage_context.get("prosecution_context") or {}
        opening = prosecution_context.get("opening_line") or prosecution_context.get("summary") or ""
        claims = (dialogue_context.get("prosecution_case") or {}).get("fixed_claim_pool") or []
        claim_summary = claims[0].get("summary", "") if claims else ""

        stage_id = ((dialogue_context.get("stage_context") or {}).get("stage_id")) or ""
        if response_mode == "defend_witness":
            if stage_id == "stage_epitaph_car":
                text = (
                    "변호인의 주장은 성립하지 않습니다! 임민수 증인의 증언은 "
                    "앤서니가 연행 호송차를 해킹했다는 검찰 주장을 뒷받침합니다."
                )
            else:
                witness_hint = witness_line[:40] + "..." if witness_line else "증인의 증언"
                text = (
                    f"변호인의 주장은 성립하지 않습니다! {witness_hint}은(는) "
                    "검찰이 제출한 핵심 증거이며, 피고인에게 불리한 사실을 명확히 보여줍니다."
                )
            tag = "basic"
        elif any(keyword in user_question for keyword in ("기소", "공소", "검찰", "유죄")):
            text = (
                f"검찰은 {claim_summary[:48]}... 이 증언이 피고인에게 불리하다고 봅니다."
                if claim_summary
                else "검찰은 제출된 증거와 증언이 피고인의 혐의를 뒷받침한다고 봅니다."
            )
            tag = "basic"
        elif opening:
            text = f"{opening[:60]}... 변호인의 질문은 아직 핵심 쟁점과 거리가 있습니다."
            tag = "think"
        elif any(keyword in user_question for keyword in ("증인", "증언", "목격")):
            text = "검찰은 이 증인의 증언이 사건의 핵심 시간대를 설명한다고 봅니다."
            tag = "basic"
        else:
            text = "검찰 입장에서 변호인의 주장은 아직 핵심 쟁점을 흔들지 못했습니다."
            tag = "basic"
        return ActorResponse(lines=[ActorLine(speaker="pros_001", dialogue=text, animation_tag=tag)])

    @staticmethod
    def undermines_witness(text: str) -> bool:
        lowered = (text or "").lower()
        return any(phrase in text or phrase in lowered for phrase in PROSECUTOR_UNDERMINE_PHRASES)

    async def generate_stage_pressure(
        self,
        *,
        user_answer: str,
        selected_evidence_ids: list[str],
        selected_evidence_details: list[dict[str, Any]],
        current_statement: Any,
        evaluation: DefenseArgumentEvaluation,
        episode: EpisodeData,
    ) -> ActorResponse:
        payload = {
            "user_answer": user_answer,
            "selected_evidence_ids": selected_evidence_ids,
            "selected_evidence_details": selected_evidence_details,
            "current_statement": current_statement.model_dump()
            if hasattr(current_statement, "model_dump")
            else current_statement,
            "evaluation": evaluation.model_dump(),
            "forbidden_claims": episode.forbidden_claims,
        }
        system = (
            f"{PROSECUTOR_BASE} 변호인의 논박에 대응하세요. "
            "변호인이 이미 말한 내용에 대해서만 반박하거나 방어하고, 아직 드러나지 않은 약점은 말하지 마세요."
        )
        return await self._generate_or_mock(
            payload,
            system,
            ProsecutorPlanMode.PRESSURE,
            self._mock_stage_pressure(evaluation, selected_evidence_ids),
        )

    async def generate_stage_interjection(
        self,
        *,
        event_type: str,
        failure_type: str | None = None,
        user_answer: str = "",
        selected_evidence_ids: list[str] | None = None,
        selected_evidence_details: list[dict[str, Any]] | None = None,
        current_statement: Any = None,
        prosecution_context: dict[str, Any] | None = None,
        evaluation: DefenseArgumentEvaluation | None = None,
        episode: EpisodeData | None = None,
    ) -> ActorResponse:
        selected_evidence_ids = selected_evidence_ids or []
        selected_evidence_details = selected_evidence_details or []
        if event_type not in {
            "stage_started",
            "irrelevant_answer",
            "no_evidence_selected",
            "witness_rescue",
            "stage_cleared",
            "irrelevant_evidence",
            "missing_core_point",
            "pure_speculation",
        }:
            return ActorResponse(lines=[])

        payload = {
            "event_type": event_type,
            "failure_type": failure_type,
            "user_answer": user_answer,
            "selected_evidence_ids": selected_evidence_ids,
            "selected_evidence_details": selected_evidence_details,
            "current_statement": current_statement.model_dump()
            if hasattr(current_statement, "model_dump")
            else current_statement,
            "prosecution_context": prosecution_context or {},
            "evaluation": evaluation.model_dump() if evaluation else None,
            "forbidden_claims": episode.forbidden_claims if episode else [],
        }
        system = (
            f"{PROSECUTOR_BASE} stage_started이면 prosecution_context를 우선 사용해 "
            "검사가 왜 이 증인을 세웠는지, 증인의 주장이 검찰 주장에 어떤 의미인지 말하세요. "
            "stage_cleared이면 해당 증언이 흔들린 점만 인정하고 다음 약점은 말하지 마세요."
        )
        return await self._generate_or_mock(
            payload,
            system,
            ProsecutorPlanMode.PRESSURE,
            self._mock_stage_interjection(event_type, failure_type, payload),
        )

    async def _generate_or_mock(
        self,
        payload: dict[str, Any],
        system: str,
        mode: ProsecutorPlanMode,
        fallback: ActorResponse,
        max_lines: int = 2,
        response_mode: str | None = None,
    ) -> ActorResponse:
        if self._use_openai:
            try:
                response = await parse_openai_structured(
                    api_key=self.openai_api_key,
                    model=self.openai_model,
                    system=system,
                    user=json.dumps(payload, ensure_ascii=False),
                    response_model=ActorResponse,
                    temperature=0.85,
                )
                return self._sanitize(response, mode, max_lines=max_lines, response_mode=response_mode)
            except Exception as e:
                logger.warning("ProsecutorActor OpenAI failed: %s", e)
        return self._sanitize(fallback, mode, max_lines=max_lines, response_mode=response_mode)

    def _default_animation(
        self,
        mode: ProsecutorPlanMode,
        response_mode: str | None,
        line_index: int,
    ) -> str:
        if mode == ProsecutorPlanMode.RETREAT:
            return "think"
        return "basic"

    def _sanitize(
        self,
        response: ActorResponse,
        mode: ProsecutorPlanMode,
        max_lines: int = 2,
        response_mode: str | None = None,
    ) -> ActorResponse:
        lines = []
        for idx, line in enumerate(response.lines[:max_lines]):
            chunks = split_dialogue_chunks(
                sanitize_character_names((line.dialogue or "").strip()),
                max_chars=MAX_DIALOGUE_CHUNK_LEN,
                max_chunks=MAX_DIALOGUE_CHUNKS_PER_LINE,
            )
            tag = (
                line.animation_tag
                if line.animation_tag not in ("idle", "normal", "basic")
                else self._default_animation(mode, response_mode, idx)
            )
            for dialogue in chunks:
                if not dialogue:
                    continue
                lines.append(
                    ActorLine(
                        speaker="pros_001",
                        dialogue=dialogue,
                        animation_tag=tag,
                    )
                )
        return ActorResponse(lines=lines) if lines else ActorResponse(lines=[])

    def _mock(self, plan: ProsecutorPlan, claim: ProsecutionClaim, evidence_details: list[dict]) -> ActorResponse:
        ev_names = ", ".join(e.get("name", "") for e in evidence_details[:2] if e.get("name"))
        if plan.mode == ProsecutorPlanMode.RETREAT:
            text = "방금 논박으로 검찰 주장 일부가 흔들린 것은 인정합니다. 하지만 검찰은 아직 입장을 철회하지 않습니다."
            tag = "think"
        else:
            suffix = f" 관련 자료는 {ev_names}입니다." if ev_names else ""
            text = f"검찰은 이 증언이 피고인의 혐의를 뒷받침한다고 봅니다. {claim.summary}{suffix}"
            tag = "basic"
        return ActorResponse(lines=[ActorLine(speaker="pros_001", dialogue=text, animation_tag=tag)])

    def _mock_stage_interjection(
        self,
        event_type: str,
        failure_type: str | None,
        payload: dict[str, Any] | None = None,
    ) -> ActorResponse:
        payload = payload or {}
        prosecution_context = payload.get("prosecution_context") or {}
        stage_id = prosecution_context.get("stage_id") or ""
        ctx = prosecution_context.get("prosecution_theory") or prosecution_context
        if event_type == "stage_started":
            if stage_id == "stage_epitaph_car":
                text = str(ctx.get("fixed_prosecutor_post_testimony_line") or "").strip()
                if not text:
                    text = (
                        "임민수 증인의 증언은 바로 이 점을 보여줍니다. "
                        "앤서니가 의도적으로 연행 호송차를 조종했다는 검찰의 주장을 확실히 뒷받침합니다."
                    )
            else:
                text = (
                    "검찰은 이 증인을 피고인의 현장 관련성을 밝히기 위해 세웠습니다. "
                    "증인은 자신이 본 내용을 말할 것입니다."
                )
            tag = "basic"
        elif event_type == "witness_rescue" and stage_id == "stage_epitaph_car":
            text = (
                "변호인의 공격은 아직 핵심을 꿰뚫지 못했습니다! "
                "임민수 증인의 증언은 앤서니가 연행 호송차를 해킹했다는 검찰 주장을 뒷받침합니다."
            )
            tag = "basic"
        elif event_type == "stage_cleared":
            text = "그 증언이 흔들린 점은 인정합니다. 그러나 검찰의 전체 주장이 끝난 것은 아닙니다."
            tag = "think"
        elif failure_type == "no_evidence_selected":
            text = "증거 없는 추측으로는 검찰 측 증언을 물리칠 수 없습니다."
            tag = "basic"
        else:
            text = "그 논박은 아직 검찰 측 증언의 힘을 충분히 꺾지 못했습니다."
            tag = "basic"
        return ActorResponse(lines=[ActorLine(speaker="pros_001", dialogue=text, animation_tag=tag)])

    def _mock_stage_pressure(
        self,
        evaluation: DefenseArgumentEvaluation,
        selected_evidence_ids: list[str],
    ) -> ActorResponse:
        if not selected_evidence_ids:
            text = "증거가 없습니다. 지금 말만으로는 검찰 측 증언을 흔들 수 없습니다."
        elif evaluation.logic_score < 0.45:
            text = "증거는 제시됐지만, 그 자료가 증언을 왜 약화시키는지 설명이 부족합니다."
        else:
            text = "논점은 보입니다. 다만 검찰은 아직 그 정도로 증언이 무너졌다고 보지 않습니다."
        return ActorResponse(lines=[ActorLine(speaker="pros_001", dialogue=text, animation_tag="basic")])
