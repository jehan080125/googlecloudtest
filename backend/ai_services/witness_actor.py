import json
import os
import re
from typing import Any, Optional

from backend.ai_services.openai_structured import parse_openai_structured
from backend.config import LLM_PROVIDER, OPENAI_WITNESS_MODEL, get_openai_api_key
from backend.core.dialogue_chunker import split_dialogue_chunks
from backend.core.korean_name_sanitizer import CHARACTER_NAME_PROMPT_RULE, sanitize_character_names
from backend.logging_config import get_logger
from backend.schemas.court import ActorLine, ActorResponse
from backend.schemas.episode import FixedWitnessTestimony
from backend.schemas.trial import AnswerVerdict, DefenseArgumentEvaluation

logger = get_logger(__name__)

MAX_DIALOGUE_CHUNK_LEN = 160
MAX_DIALOGUE_CHUNKS_PER_LINE = 4


class WitnessActorLLM:
    def __init__(self, api_key: Optional[str] = None):
        self.openai_api_key = get_openai_api_key("witness", api_key)
        self.openai_model = OPENAI_WITNESS_MODEL
        disable_config_llm = api_key is None and bool(os.getenv("PYTEST_CURRENT_TEST"))
        self._use_openai = (
            not disable_config_llm
            and bool(self.openai_api_key)
            and LLM_PROVIDER in ("auto", "openai")
        )

    async def speak_testimony(
        self, testimony: FixedWitnessTestimony, witness_id: str, witness_name: str = "증인"
    ) -> ActorResponse:
        # Fixed testimony must be delivered verbatim from episode JSON (never LLM).
        dialogue = sanitize_character_names(str(getattr(testimony, "text", "") or ""))
        return ActorResponse(
            lines=[ActorLine(speaker=witness_id, dialogue=dialogue, animation_tag="idle")]
        )

    async def speak_retreat(self, witness_id: str, success: bool = True) -> ActorResponse:
        payload = {"success": success, "witness_id": witness_id}
        system = (
            "당신은 논박을 당한 증인입니다. 변호인의 지적 때문에 방금 한 말이 흔들린 상황입니다. "
            "완전히 자백하지는 말고, 당황하거나 말을 고르는 반응을 하세요. "
            "새로운 증거·사실은 만들지 마세요. 한국어 1줄, 80자 이하. "
            f"JSON lines만 반환하세요. {CHARACTER_NAME_PROMPT_RULE}"
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
                        temperature=0.9,
                    ),
                    "retreat",
                    witness_id,
                    None,
                    70,
                )
            except Exception as e:
                logger.warning("Witness retreat OpenAI failed: %s", e)

        text = "그, 그건... 제가 착각했을 수도 있습니다. 하지만 제 말이 전부 틀렸다는 건 아닙니다."
        return ActorResponse(lines=[ActorLine(speaker=witness_id, dialogue=text, animation_tag="sweat")])

    async def respond_to_free_question(
        self,
        *,
        user_question: str,
        witness_id: str,
        current_statement: dict[str, Any] | None,
        witness_mental: int,
        dialogue_context: dict[str, Any] | None = None,
        verifier_feedback: str | None = None,
    ) -> ActorResponse:
        dialogue_context = dialogue_context or {}
        character = dialogue_context.get("character") or {}
        response_mode = dialogue_context.get("response_mode", "answer_player")
        max_lines = dialogue_context.get("max_response_lines", 1)
        payload = {
            "role": dialogue_context.get("role", "witness"),
            "character": character,
            "user_question": user_question,
            "response_mode": response_mode,
            "turn_batch_lines": dialogue_context.get("turn_batch_lines") or [],
            "batch_position": dialogue_context.get("batch_position", 0),
            "witness_id": witness_id,
            "current_statement": current_statement or {},
            "stage_context": dialogue_context.get("stage_context") or {},
            "dialogue_history": dialogue_context.get("dialogue_history") or [],
            "character_knowledge": dialogue_context.get("character_knowledge") or {},
            "case_summary": dialogue_context.get("case_summary") or {},
            "inventory_evidence": dialogue_context.get("inventory_evidence") or [],
            "allowed_lies": dialogue_context.get("allowed_lies") or [],
            "exchange_count": dialogue_context.get("exchange_count", 0),
            "witness_mental": witness_mental,
            "witness_emotion_band": self._emotion_band(witness_mental),
            "forbidden_claims": dialogue_context.get("forbidden_claims") or [],
            "verifier_feedback": verifier_feedback or "",
        }
        response_mode = dialogue_context.get("response_mode", "answer_player")
        max_lines = dialogue_context.get("max_response_lines", 1)
        if response_mode == "followup_after_prosecutor":
            mode_instruction = (
                "검사가 방금 당신의 증언을 옹호했습니다(turn_batch_lines 참고). "
                "그 옹호에 이어 한 줄로 당신의 증언을 다시 단호히 확인하세요. "
                "긴장했지만 자신의 말을 지키는 톤으로, 검사의 말에 동의하며 버티세요."
            )
        else:
            mode_instruction = (
                "변호인의 자유 질문(user_question)에 답하세요. "
                "자신의 증언(current_statement, own_statements)을 방어하고, "
                "알고 있는 범위(character_knowledge.scope) 안에서만 답하세요. "
                "allowed_lies 범위 안에서만 말을 바꿀 수 있습니다."
            )
        system = (
            f"당신은 법정 게임의 증인 '{character.get('name', '증인')}'입니다. "
            f"{mode_instruction} "
            "character, character_knowledge, case_summary, stage_context, dialogue_history, "
            "turn_batch_lines, allowed_lies를 참고하세요. "
            "이전 대화 맥락(dialogue_history)과 같은 턴의 선행 대사(turn_batch_lines)를 반드시 이어가세요. "
            "새 사실·새 증거·자백은 만들지 마세요. "
            f"{self._trial_specific_guardrail(dialogue_context, current_statement)} "
            f"{self._witness_specific_guardrail(dialogue_context)} "
            f"{self._forbidden_claims_instruction(payload['forbidden_claims'])} "
            f"{self._verification_feedback_instruction(verifier_feedback)} "
            "법정 드라마 톤으로 짧고 긴장감 있는 대사를 쓰세요. "
            f"한국어 1~{max_lines}줄, 각 줄 100자 이하. "
            "animation_tag 기본값은 basic(또는 idle)이며, 당황·압박·붕괴 등 감정이 고조된 순간에만 "
            "embarrassed, sweat, shaken, breakdown을 쓰세요. "
            f"JSON lines만 반환하세요. {CHARACTER_NAME_PROMPT_RULE}"
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
                        temperature=0.85,
                    ),
                    "free_question",
                    witness_id,
                    None,
                    witness_mental,
                    max_lines=max_lines,
                )
            except Exception as e:
                logger.warning("Witness free question OpenAI failed: %s", e)

        mock = self._mock_free_question(
            user_question=user_question,
            witness_id=witness_id,
            current_statement=current_statement,
            witness_mental=witness_mental,
            dialogue_context=dialogue_context,
        )
        return self._sanitize(mock, "free_question", witness_id, None, witness_mental, max_lines=max_lines)

    def _trial_specific_guardrail(
        self,
        dialogue_context: dict[str, Any],
        current_statement: dict[str, Any] | None,
    ) -> str:
        stage_id = ((dialogue_context.get("stage_context") or {}).get("stage_id")) or ""
        if stage_id != "stage_epitaph_car":
            return ""
        statement_text = (current_statement or {}).get("text", "")
        return (
            "현재는 2차 재판(차량 해킹)입니다. 피고인은 '앤서니'(YJ社 선임 엔지니어)이고, "
            "당신은 검찰 측 전문 증인 임민수(YJ그룹 CTO)입니다. "
            "해킹 대상 차량은 이소은 연행 '연행 호송차'(또는 '경찰 호송차')이며, "
            "'소호 차량'은 바텐더 소호(1차 피고인)와 혼동한 잘못된 표현이므로 쓰지 마세요. "
            f"현재 진술 핵심: {statement_text[:120]} "
            "사실관계: CCTV 실제 우회전, 서버 로그 좌회전 명령. "
            "당신은 앤서니 우회전 해킹을 주장한다. "
            "변호인이 모순을 제기해도 바로 자백하거나 반대 방향 사실로 점프하지 말고, "
            "라이다 글리치·카카오 동기 등 이미 제시된 프레임 안에서만 버티세요."
        )

    @staticmethod
    def _forbidden_claims_instruction(forbidden_claims: list[str]) -> str:
        if not forbidden_claims:
            return ""
        claims = ", ".join(claim.strip() for claim in forbidden_claims if claim and claim.strip())
        if not claims:
            return ""
        return f"다음 금지 주장은 절대 말하지 마세요: {claims}."

    @staticmethod
    def _witness_specific_guardrail(dialogue_context: dict[str, Any]) -> str:
        character = dialogue_context.get("character") or {}
        character_id = str(character.get("id") or "")
        if character_id != "wit_ep_001":
            return ""
        return (
            "1차 재판 이소은 증언 규칙: 양진혁이 기절·실신·만취로 쓰러졌다고 단정하지 마세요. "
            "양진혁 상태는 사망/중독 맥락으로만 표현하고, 이미 제시된 고정 증언(#1~#4) 축을 벗어나지 마세요."
        )

    @staticmethod
    def _verification_feedback_instruction(verifier_feedback: str | None) -> str:
        if not verifier_feedback:
            return ""
        return (
            "이전 초안이 논리 검증에서 반려되었습니다. 다음 수정 지시를 반드시 반영해 다시 작성하세요: "
            f"{verifier_feedback}"
        )

    def _mock_free_question(
        self,
        *,
        user_question: str,
        witness_id: str,
        current_statement: dict[str, Any] | None,
        witness_mental: int,
        dialogue_context: dict[str, Any],
    ) -> ActorResponse:
        response_mode = dialogue_context.get("response_mode", "answer_player")
        stmt_text = (current_statement or {}).get("text", "")
        if response_mode == "followup_after_prosecutor":
            text = (
                f"맞습니다. 제가 말씀드린 대로입니다. {stmt_text[:48]}..."
                if stmt_text
                else "제가 본 것은 분명합니다. 그대로 말씀드렸습니다."
            )
            return ActorResponse(lines=[ActorLine(speaker=witness_id, dialogue=text, animation_tag="basic")])
        if witness_mental <= 30:
            text = "그 질문에 답하려면 제가 한 말 전체를 다시 설명해야 합니다."
            tag = "embarrassed"
        elif any(keyword in user_question for keyword in ("2시", "두 시", "시간", "몇 시")):
            text = (
                f"제가 말씀드린 대로 오후 2시에 신고했습니다. {stmt_text[:36]}..."
                if stmt_text
                else "제가 본 시간은 그때 그대로입니다."
            )
            tag = "basic"
        elif any(keyword in user_question for keyword in ("부검", "사망", "시체")):
            text = "저는 의학적 기록까지는 모릅니다. 저는 그때 방에서 본 것만 말하고 있습니다."
            tag = "basic"
        elif any(keyword in user_question for keyword in ("증인", "봤", "목격", "방")):
            text = (
                stmt_text[:72]
                if stmt_text
                else "제가 본 것만 말씀드렸습니다. 그 이상은 확실하지 않습니다."
            )
            tag = "basic"
        else:
            text = "제가 본 것만 말씀드렸습니다. 그 이상은 확실하지 않습니다."
            tag = "basic"
        return ActorResponse(lines=[ActorLine(speaker=witness_id, dialogue=text, animation_tag=tag)])

    async def generate_stage_reaction(
        self,
        *,
        event_type: str,
        witness_id: str,
        evaluation: DefenseArgumentEvaluation | dict[str, Any] | None,
        current_statement: dict[str, Any] | None,
        user_answer: str,
        selected_evidence_ids: list[str],
        witness_mental: int,
        stage_result: dict[str, Any] | None = None,
        next_counter_statement: dict[str, Any] | None = None,
    ) -> ActorResponse:
        if (
            event_type == "witness_counter"
            and next_counter_statement
            and bool(next_counter_statement.get("is_fixed", False))
        ):
            fixed_text = str(next_counter_statement.get("text") or "")
            return ActorResponse(
                lines=[ActorLine(speaker=witness_id, dialogue=fixed_text, animation_tag="embarrassed")]
            )

        parsed_evaluation = self._coerce_evaluation(evaluation)
        payload = {
            "event_type": event_type,
            "evaluation": parsed_evaluation.model_dump() if parsed_evaluation else evaluation,
            "current_statement": current_statement or {},
            "user_answer": user_answer,
            "selected_evidence_ids": selected_evidence_ids,
            "witness_mental": witness_mental,
            "witness_emotion_band": self._emotion_band(witness_mental),
            "stage_result": stage_result or {},
            "next_counter_statement": next_counter_statement or {},
        }

        if event_type == "witness_counter" and next_counter_statement:
            system = (
                "당신은 궁지에 몰린 증인입니다. 변호인의 논박을 잠깐 인정하거나 회피한 뒤, "
                "next_counter_statement.text의 새 주장을 반드시 말하세요. 핵심 의미를 바꾸지 말고, "
                "증인이 스스로 말을 바꿔 버티는 느낌이 나야 합니다. 새 사실은 만들지 마세요. "
                f"한국어 1~2줄, 각 줄 90자 이하. JSON lines만 반환하세요. {CHARACTER_NAME_PROMPT_RULE}"
            )
        else:
            system = (
                "당신은 법정 게임의 증인입니다. current_statement와 evaluation 범위 안에서만 반응하세요. "
                "변호인이 무엇을 찔렀는지 이해한 듯, 당황·반발·허세·붕괴를 상황에 맞게 표현하세요. "
                f"새 증언, 새 증거, 자백을 만들지 마세요. 한국어 1줄, 90자 이하. "
                f"JSON lines만 반환하세요. {CHARACTER_NAME_PROMPT_RULE}"
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
                        temperature=0.9,
                    ),
                    event_type,
                    witness_id,
                    parsed_evaluation,
                    witness_mental,
                )
            except Exception as e:
                logger.warning("WitnessActor OpenAI failed: %s", e)

        if event_type == "witness_counter" and next_counter_statement:
            text = next_counter_statement.get("text", "")
            return ActorResponse(lines=[ActorLine(speaker=witness_id, dialogue=text, animation_tag="embarrassed")])
        return self._mock_stage_reaction(event_type, witness_id, parsed_evaluation, witness_mental)

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
        witness_id: str,
        evaluation: DefenseArgumentEvaluation | None,
        witness_mental: int,
        max_lines: int = 2,
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
                if line.animation_tag not in ("idle", "normal", "basic") or idx == 0
                else self._animation_for(event_type, evaluation, witness_mental)
            )
            for chunk in chunks:
                dialogue = self._sanitize_victim_fainting_claim(chunk)
                if not dialogue:
                    continue
                lines.append(
                    ActorLine(
                        speaker=witness_id,
                        dialogue=dialogue,
                        animation_tag=tag,
                    )
                )
        if lines:
            return ActorResponse(lines=lines)
        return self._mock_stage_reaction(event_type, witness_id, evaluation, witness_mental)

    @staticmethod
    def _sanitize_victim_fainting_claim(dialogue: str) -> str:
        if not dialogue or "양진혁" not in dialogue:
            return dialogue
        updated = dialogue
        # Victim fainting is disallowed in this episode; reframe as death.
        updated = re.sub(r"양진혁(이|가|은|는)?\s*(기절|실신)\w*", "양진혁은 사망", updated)
        updated = re.sub(r"양진혁가\s*(기절|실신)\w*", "양진혁은 사망", updated)
        updated = re.sub(r"양진혁(이|가|은|는)?\s*취\w*\s*쓰러\w*", "양진혁은 쓰러져 사망", updated)
        updated = updated.replace("양진혁이 쓰러졌다", "양진혁이 쓰러져 사망했다")
        updated = updated.replace("양진혁가 쓰러졌다", "양진혁이 쓰러져 사망했다")
        updated = updated.replace("양진혁은 쓰러졌다", "양진혁은 쓰러져 사망했다")
        return updated

    def _mock_stage_reaction(
        self,
        event_type: str,
        witness_id: str,
        evaluation: DefenseArgumentEvaluation | None,
        witness_mental: int,
    ) -> ActorResponse:
        verdict = evaluation.verdict if evaluation else None
        if event_type == "witness_breakdown" or witness_mental <= 0:
            text = "아니야... 그럴 리가 없어... 내가 그런 걸 어떻게 알았다는 거야..."
        elif witness_mental <= 30:
            text = "그만하십시오! 제가 거짓말만 한 사람처럼 몰아가지 마세요!"
        elif verdict == AnswerVerdict.SUCCESS:
            text = "그, 그건... 잠깐 착각했을 뿐입니다. 제 말에는 아직 설명이 됩니다!"
        elif verdict == AnswerVerdict.PARTIAL_SUCCESS:
            text = "그 정도로 제 증언 전체가 틀렸다고 할 수는 없습니다."
        else:
            text = "변호인님, 말만 그럴듯하지 증거와는 맞지 않습니다."

        return ActorResponse(
            lines=[
                ActorLine(
                    speaker=witness_id,
                    dialogue=text,
                    animation_tag=self._animation_for(event_type, evaluation, witness_mental),
                )
            ]
        )

    def _emotion_band(self, witness_mental: int) -> str:
        if witness_mental <= 0:
            return "breakdown"
        if witness_mental <= 30:
            return "cornered"
        if witness_mental <= 65:
            return "shaken"
        return "annoyed"

    def _animation_for(
        self,
        event_type: str,
        evaluation: DefenseArgumentEvaluation | None,
        witness_mental: int,
    ) -> str:
        if event_type == "witness_breakdown" or witness_mental <= 0:
            return "breakdown"
        if event_type == "witness_shaken":
            return "shaken" if witness_mental > 30 else "embarrassed"
        if event_type == "witness_counter":
            return "embarrassed"
        if witness_mental <= 30:
            return "embarrassed"
        if evaluation:
            if evaluation.verdict == AnswerVerdict.SUCCESS:
                return "embarrassed"
            if evaluation.verdict == AnswerVerdict.PARTIAL_SUCCESS:
                return "sweat"
            if evaluation.verdict == AnswerVerdict.FAIL:
                return "basic"
        return "basic"
