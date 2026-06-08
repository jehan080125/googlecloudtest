import json
import os
from typing import Any, Optional

from pydantic import BaseModel, Field

from backend.ai_services.judge_actor import WITNESS_VOICE_MARKERS
from backend.ai_services.openai_structured import parse_openai_structured
from backend.config import LLM_PROVIDER, OPENAI_VERIFIER_MODEL, get_openai_api_key
from backend.logging_config import get_logger
from backend.schemas.court import ActorLine, ActorResponse

logger = get_logger(__name__)

_PROSECUTOR_UNDERMINE_PHRASES = (
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


class VerificationResult(BaseModel):
    valid: bool = True
    issues: list[str] = Field(default_factory=list)
    suggested_fix: str = ""


class ResponseVerifierLLM:
    """Logical consistency guardrail for courtroom actor responses."""

    def __init__(self, api_key: Optional[str] = None):
        self.openai_api_key = get_openai_api_key("system", api_key)
        self.openai_model = OPENAI_VERIFIER_MODEL
        disable_config_llm = api_key is None and bool(os.getenv("PYTEST_CURRENT_TEST"))
        llm_verifier_enabled = os.getenv("ENABLE_LLM_VERIFIER", "true").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.openai_timeout_s = float(os.getenv("OPENAI_VERIFIER_TIMEOUT_SEC", "6"))
        self._use_openai = (
            not disable_config_llm
            and llm_verifier_enabled
            and bool(self.openai_api_key)
            and LLM_PROVIDER in ("auto", "openai")
        )

    async def verify(
        self,
        *,
        role: str,
        stage_id: str,
        stage_phase: str,
        response: ActorResponse,
        user_text: str,
        current_statement: dict[str, Any] | None,
        turn_batch_lines: list[dict[str, Any]],
        forbidden_claims: list[str] | None = None,
        inventory_evidence: list[dict[str, Any]] | None = None,
    ) -> VerificationResult:
        heuristic = self._heuristic_verify(
            role=role,
            stage_id=stage_id,
            response=response,
            current_statement=current_statement,
            turn_batch_lines=turn_batch_lines,
            forbidden_claims=forbidden_claims or [],
        )
        if not heuristic.valid:
            return heuristic

        if not self._use_openai:
            return heuristic

        payload = {
            "role": role,
            "stage_id": stage_id,
            "stage_phase": stage_phase,
            "user_text": user_text,
            "response": response.model_dump(),
            "current_statement": current_statement or {},
            "turn_batch_lines": turn_batch_lines,
            "forbidden_claims": forbidden_claims or [],
            "inventory_evidence": inventory_evidence or [],
        }
        system = (
            "당신은 법정 대사 검증기입니다. 출력 JSON 형식: {valid, issues, suggested_fix}.\n"
            "검증 기준:\n"
            "1) role이 prosecutor면 같은 턴의 증인 발언(turn_batch_lines)을 약화·부정하면 invalid.\n"
            "2) role이 witness면 current_statement의 핵심 주장 축을 임의로 뒤집으면 invalid.\n"
            "3) role이 judge면 새 사실을 창작하거나 한쪽 편을 들어 단정하면 invalid. "
            "검찰·증인 옹호('일관적이니 괜찮', '증인 발언은 신뢰')나 "
            "변호인 이의를 근거 없이 일괄 기각하면 invalid.\n"
            "4) 금지 주장(forbidden_claims)을 말하면 invalid.\n"
            "5) 응답이 사건 맥락과 무관하거나 자기모순이면 invalid.\n"
            "valid=false일 때 suggested_fix는 1문장 한국어 지시문으로 작성하세요."
        )
        try:
            return await parse_openai_structured(
                api_key=self.openai_api_key,
                model=self.openai_model,
                system=system,
                user=json.dumps(payload, ensure_ascii=False),
                response_model=VerificationResult,
                temperature=0.1,
                timeout_s=self.openai_timeout_s,
            )
        except Exception as exc:
            logger.warning("Response verifier OpenAI failed: %s", exc)
            return heuristic

    def _heuristic_verify(
        self,
        *,
        role: str,
        stage_id: str,
        response: ActorResponse,
        current_statement: dict[str, Any] | None,
        turn_batch_lines: list[dict[str, Any]],
        forbidden_claims: list[str],
    ) -> VerificationResult:
        issues: list[str] = []
        lines: list[ActorLine] = response.lines or []
        if not lines:
            issues.append("응답 줄이 비어 있습니다.")
            return VerificationResult(valid=False, issues=issues, suggested_fix="한 줄 이상 분명하게 답하세요.")

        merged = " ".join((line.dialogue or "").strip() for line in lines).strip()
        lowered = merged.lower()
        if not merged:
            issues.append("대사가 비어 있습니다.")

        for forbidden in forbidden_claims:
            if forbidden and forbidden.lower() in lowered:
                issues.append(f"금지 주장 유출: {forbidden}")

        if role == "prosecutor":
            if any(phrase in merged or phrase in lowered for phrase in _PROSECUTOR_UNDERMINE_PHRASES):
                issues.append("검사가 증인을 스스로 약화했습니다.")
            witness_lines = [entry.get("text", "") for entry in turn_batch_lines if "wit" in entry.get("speaker", "")]
            if witness_lines and any(keyword in merged for keyword in ("제가 틀렸", "증언이 틀", "못 믿")):
                issues.append("검사가 같은 턴 증언을 부정했습니다.")
            if stage_id == "stage_epitaph_car" and any(
                term in merged for term in ("소호 차량", "소호차량", "소호의 차량")
            ):
                issues.append("검사가 2차 사건 차량을 바텐더 소호의 차량으로 잘못 지칭했습니다.")

        if role == "witness" and current_statement:
            statement = current_statement.get("text", "")
            if stage_id == "stage_epitaph_car":
                if "좌회전" in statement and "우회전" in merged and "모순" not in merged:
                    issues.append("증인이 핵심 주장(좌회전)을 임의로 뒤집었습니다.")
                if "우회전" in statement and "좌회전" in merged and "모순" not in merged:
                    issues.append("증인이 핵심 주장(우회전)을 임의로 뒤집었습니다.")

        if role == "judge":
            prosecution_bias_phrases = (
                "검찰이 확실히 옳",
                "변호인이 확실히 틀",
                "이미 확정",
                "일관적이니 괜찮",
                "증인 발언이 일관",
                "증언은 신뢰할 만",
                "검찰 주장이 타당",
                "검사 말이 맞",
            )
            if any(term in merged for term in prosecution_bias_phrases):
                issues.append("판사가 중립성을 벗어난 단정 표현을 사용했습니다.")
            if self._sounds_like_witness_voice(merged):
                issues.append("판사가 증인 1인칭 대사를 사용했습니다.")

        if not issues:
            return VerificationResult(valid=True, issues=[], suggested_fix="")
        suggested_fix = (
            "중립을 유지하고 논리적 일관성만 평가하세요. 증인 1인칭·한쪽 편 든 표현을 쓰지 마세요."
            if role == "judge"
            else "역할 제약을 지키고 현재 증언의 핵심 주장만 일관되게 방어하세요."
        )
        return VerificationResult(valid=False, issues=issues, suggested_fix=suggested_fix)

    @staticmethod
    def _sounds_like_witness_voice(dialogue: str) -> bool:
        text = (dialogue or "").strip()
        if not text:
            return False
        return any(marker in text for marker in WITNESS_VOICE_MARKERS)
