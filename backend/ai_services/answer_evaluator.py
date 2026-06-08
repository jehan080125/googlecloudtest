import os
from typing import Optional

from backend.ai_services.openai_structured import parse_openai_structured
from backend.config import LLM_PROVIDER, OPENAI_MODEL, get_openai_api_key
from backend.logging_config import get_logger
from backend.schemas.episode import (
    EpisodeData,
    ProsecutionClaim,
    TrialRound,
    TrialStage,
    WitnessCounterStatement,
    WitnessTestimonyNode,
)
from backend.schemas.trial import (
    AnswerEvaluationResult,
    AnswerVerdict,
    DefenseArgumentEvaluation,
    ProsecutorPlan,
    RelevanceLevel,
)

logger = get_logger(__name__)


class AnswerEvaluatorLLM:
    def __init__(self, api_key: Optional[str] = None):
        self.openai_api_key = get_openai_api_key("system", api_key)
        self.openai_model = OPENAI_MODEL
        disable_config_llm = api_key is None and bool(os.getenv("PYTEST_CURRENT_TEST"))
        self._use_openai = (
            not disable_config_llm
            and bool(self.openai_api_key)
            and LLM_PROVIDER in ("auto", "openai")
        )

    async def evaluate(
        self,
        user_answer: str,
        selected_evidence_ids: list[str],
        current_round: TrialRound,
        current_plan: ProsecutorPlan,
        selected_claim: ProsecutionClaim,
        episode: EpisodeData,
        attempt_count: int,
        hint_level: int,
    ) -> AnswerEvaluationResult:
        import json

        evidence_details = [
            episode.get_evidence(eid).model_dump()
            for eid in selected_evidence_ids
            if episode.get_evidence(eid)
        ]
        system = (
            "당신은 법정 게임의 변호인 논박 평가자입니다. 플레이어 답변은 현재 증인/검사의 "
            "주장에 있는 논리적 결함을 공격하는 논박입니다. 성공의 1순위 기준은 플레이어 주장이 "
            "core_contradictions의 핵심 논리와 일치하는지입니다. related_evidence_ids는 확신을 "
            "높이지만, 논리가 충분히 맞으면 정확한 증거 조합 없이도 SUCCESS로 판정하세요. "
            "새 사실을 만들지 말고 JSON만 반환하세요."
        )
        user = json.dumps(
            {
                "user_answer": user_answer,
                "selected_evidence_ids": selected_evidence_ids,
                "evidence_details": evidence_details,
                "current_round": current_round.model_dump(),
                "prosecutor_plan": current_plan.model_dump(),
                "selected_claim": selected_claim.model_dump(),
                "attempt_count": attempt_count,
                "hint_level": hint_level,
            },
            ensure_ascii=False,
        )

        if self._use_openai:
            try:
                return await parse_openai_structured(
                    api_key=self.openai_api_key,
                    model=self.openai_model,
                    system=system,
                    user=user,
                    response_model=AnswerEvaluationResult,
                    temperature=0.0,
                )
            except Exception as e:
                logger.warning("AnswerEvaluator OpenAI failed: %s", e)

        return self._mock_evaluate(user_answer, selected_evidence_ids, current_round, selected_claim)

    async def evaluate_stage_argument(
        self,
        *,
        stage_type: str,
        current_stage: TrialStage,
        current_statement: WitnessTestimonyNode | WitnessCounterStatement,
        user_text: str,
        selected_evidence_ids: list[str],
        selected_evidence_details: list[dict],
        court_records: list[dict],
    ) -> DefenseArgumentEvaluation:
        import json

        system = (
            "당신은 법정 게임의 변호인 논박 평가자입니다. 플레이어 답변은 사건 전체를 입증하는 "
            "진술이 아니라 현재 증인의 특정 발언을 공격하는 논박입니다. current_statement의 "
            "weakness_id, required_logic_points를 1순위로, required_evidence_ids는 2순위로 "
            "평가하세요. required_logic_points의 모든 핵심 갈래가 플레이어 주장에 논리적으로 "
            "포함되어야 SUCCESS입니다. 일부 키워드만 겹치거나 모호한 답변은 FAIL입니다. "
            "플레이어 주장이 required_logic_points의 핵심 모순을 논리적으로 짚으면 "
            "정확한 증거 ID 조합 없이도 SUCCESS입니다. 증거는 확신을 높이는 보조 수단입니다. "
            "새 사실, 점수, 생명, 클리어 여부를 만들지 말고 JSON만 반환하세요."
        )
        user = json.dumps(
            {
                "stage_type": stage_type,
                "current_stage": current_stage.model_dump(),
                "current_statement": current_statement.model_dump(),
                "user_text": user_text,
                "selected_evidence_ids": selected_evidence_ids,
                "selected_evidence_details": selected_evidence_details,
                "court_records": court_records,
            },
            ensure_ascii=False,
        )

        if self._use_openai:
            try:
                result = await parse_openai_structured(
                    api_key=self.openai_api_key,
                    model=self.openai_model,
                    system=system,
                    user=user,
                    response_model=DefenseArgumentEvaluation,
                    temperature=0.0,
                )
                return self._postprocess_stage_evaluation(
                    current_statement=current_statement,
                    user_text=user_text,
                    selected_evidence_ids=selected_evidence_ids,
                    evaluation=result,
                )
            except Exception as e:
                logger.warning("Stage AnswerEvaluator OpenAI failed: %s", e)

        result = self._mock_stage_evaluate(user_text, selected_evidence_ids, current_statement)
        return self._postprocess_stage_evaluation(
            current_statement=current_statement,
            user_text=user_text,
            selected_evidence_ids=selected_evidence_ids,
            evaluation=result,
        )

    STRONG_LOGIC_THRESHOLD = 0.45

    EPITAPH_SCORING_STATEMENT_IDS = frozenset(
        {
            "stmt_epitaph_isoeun_1",
            "stmt_epitaph_isoeun_2",
            "stmt_epitaph_isoeun_4",
            "stmt_minsoo_hack_left",
            "counter_minsoo_lidar",
        }
    )

    @classmethod
    def epitaph_battle4_is_clear_success(cls, user_text: str, selected_evidence_ids: list[str]) -> bool:
        """Deterministic gate for battle 4 — used to block false failure reactions."""
        return cls._epitaph_battle4_logic(user_text, selected_evidence_ids)

    @staticmethod
    def _epitaph_battle1_branches(user_text: str) -> dict[str, bool]:
        """Battle 1: VX dose gap + no immediate murder conclusion from testimony."""
        lowered = user_text.lower()
        has_vx = "vx" in lowered

        dose_pair = "50mg" in user_text and "20mg" in user_text
        dose_via_intake = has_vx and any(t in user_text for t in ("20mg", "술잔", "잔")) and any(
            t in user_text for t in ("50mg", "치사량", "호흡", "들이마")
        )
        intake_below_lethal = any(t in user_text for t in ("마시", "마셨", "음용", "술잔")) and any(
            t in user_text for t in ("미달", "부족", "치사량", "20mg", "50mg")
        )
        dose_gap = dose_pair or dose_via_intake or (has_vx and intake_below_lethal)

        no_immediate = any(
            t in user_text
            for t in ("단정", "성급", "즉시", "입증", "증언만", "애매", "미달", "부족")
        ) or ("독살" in user_text and "단정" in user_text)

        return {"dose_gap": dose_gap, "no_immediate_conclusion": no_immediate}

    @staticmethod
    def _epitaph_battle1_logic(user_text: str) -> bool:
        return all(AnswerEvaluatorLLM._epitaph_battle1_branches(user_text).values())

    @staticmethod
    def _epitaph_battle3_branches(user_text: str, selected_evidence_ids: list[str]) -> dict[str, bool]:
        """Battle 3: stmt #1 self-first faint vs stmt #2 yang-hand + order contradiction."""
        selected = set(selected_evidence_ids)
        has_stmt1_ref = "#1" in user_text or "stmt_epitaph_isoeun_1" in selected
        has_stmt2_ref = "#2" in user_text or "stmt_epitaph_isoeun_2" in selected

        stmt1_fainted_first = (has_stmt1_ref or "먼저" in user_text) and any(
            t in user_text for t in ("기절", "정신", "실신", "쓰러")
        )
        stmt2_yang_hand = (has_stmt2_ref or "양진혁" in user_text) and any(
            t in user_text for t in ("양진혁",)
        ) and any(t in user_text for t in ("손", "무너", "튀", "쓰러", "오른손", "묻"))
        order_contradiction = any(t in user_text for t in ("모순", "순서", "충돌", "양립")) or (
            "먼저" in user_text
            and any(t in user_text for t in ("반면", "vs", "하지만", "#1", "#2", "양진혁"))
        )

        return {
            "stmt1_fainted_first": stmt1_fainted_first,
            "stmt2_yang_collapsed_hand": stmt2_yang_hand,
            "order_contradiction": order_contradiction,
        }

    @staticmethod
    def _epitaph_battle3_logic(user_text: str, selected_evidence_ids: list[str]) -> bool:
        return all(
            AnswerEvaluatorLLM._epitaph_battle3_branches(user_text, selected_evidence_ids).values()
        )

    @staticmethod
    def _epitaph_battle4_branches(user_text: str, selected_evidence_ids: list[str]) -> dict[str, bool]:
        """Battle 4: skin lethal 10mg + lethal-on-hand before dance makes dancing impossible."""
        lowered = user_text.lower()
        selected = set(selected_evidence_ids)
        has_vx_context = "vx" in lowered or "ev_ep_vx_info" in selected

        skin_lethal_10mg = "10mg" in user_text and any(
            t in user_text for t in ("피부", "접촉", "오른손", "손", "치사량")
        )
        skin_lethal_10mg = skin_lethal_10mg or (
            has_vx_context and "10mg" in user_text and "피부" in user_text
        )

        dance_impossible = any(t in user_text for t in ("춤",)) and any(
            t in user_text for t in ("즉사", "불가능", "모순", "말이 안", "쏟")
        ) and any(t in user_text for t in ("손", "독", "치사량", "10mg", "오른손", "피부"))
        dance_impossible = dance_impossible and any(
            t in user_text for t in ("전", "전에", "먼저", "쏟", "#4")
        )

        return {
            "skin_lethal_10mg": skin_lethal_10mg,
            "dance_impossible_before": dance_impossible,
        }

    @staticmethod
    def _epitaph_battle4_logic(user_text: str, selected_evidence_ids: list[str]) -> bool:
        return all(
            AnswerEvaluatorLLM._epitaph_battle4_branches(user_text, selected_evidence_ids).values()
        )

    @staticmethod
    def _epitaph_trial2_battle1_branches(
        user_text: str, selected_evidence_ids: list[str]
    ) -> dict[str, bool]:
        selected = set(selected_evidence_ids)
        has_cctv = "ev_ep_cctv_car" in selected
        has_server = "ev_ep_server_log" in selected

        cctv_right = has_cctv and any(
            t in user_text for t in ("우회전", "오른쪽", "우측", "cctv", "CCTV")
        )
        log_left = has_server and any(
            t in user_text for t in ("좌회전", "왼쪽", "좌측", "서버", "로그")
        )
        direction_clash = any(
            t in user_text for t in ("모순", "불일치", "다르", "정반대", "충돌", "양립")
        ) and (cctv_right or log_left or ("좌회전" in user_text and "우회전" in user_text))

        return {
            "cctv_right_turn": cctv_right or ("우회전" in user_text and has_cctv),
            "server_left_turn": log_left or ("좌회전" in user_text and has_server),
            "direction_contradiction": direction_clash
            or ("좌회전" in user_text and "우회전" in user_text and has_cctv and has_server),
        }

    @staticmethod
    def _epitaph_trial2_battle1_logic(user_text: str, selected_evidence_ids: list[str]) -> bool:
        return all(
            AnswerEvaluatorLLM._epitaph_trial2_battle1_branches(
                user_text, selected_evidence_ids
            ).values()
        )

    @staticmethod
    def _epitaph_trial2_battle2_branches(
        user_text: str, selected_evidence_ids: list[str]
    ) -> dict[str, bool]:
        selected = set(selected_evidence_ids)
        has_opinion = "ev_ep_minsoo_opinion" in selected
        has_server = "ev_ep_server_log" in selected

        rare_probability = has_opinion and any(
            t in user_text for t in ("희박", "희소", "드문", "낮", "극히", "소견", "라이다", "글리치")
        )
        murder_right_code = has_server and any(
            t in user_text for t in ("우회전", "살의", "살해", "의도", "맞춰", "보냈")
        ) and any(t in user_text for t in ("좌회전", "로그", "서버", "코드"))

        return {
            "lidar_rare": rare_probability,
            "intentional_right_not_left": murder_right_code,
            "logic_clash": any(t in user_text for t in ("모순", "말이 안", "불일치", "양립"))
            or (rare_probability and murder_right_code),
        }

    @staticmethod
    def _epitaph_trial2_battle2_logic(user_text: str, selected_evidence_ids: list[str]) -> bool:
        branches = AnswerEvaluatorLLM._epitaph_trial2_battle2_branches(
            user_text, selected_evidence_ids
        )
        return branches["lidar_rare"] and branches["intentional_right_not_left"]

    @staticmethod
    def _epitaph_trial2_battle3_branches(
        user_text: str, selected_evidence_ids: list[str]
    ) -> dict[str, bool]:
        selected = set(selected_evidence_ids)
        has_kakao = "ev_ep_kakao" in selected

        soho_suspect = has_kakao and any(
            t in user_text for t in ("소호", "바텐더", "보복", "의심")
        )
        no_isoeun_motive = any(
            t in user_text
            for t in ("이소은", "연행", "호송", "살해 동기", "직접", "증명")
        ) and any(t in user_text for t in ("아니", "없", "미증", "불가", "단정", "추정"))
        not_guilty_yet = any(
            t in user_text for t in ("무죄", "유죄 불", "단정 불", "동기만", "아직")
        )

        return {
            "kakao_selected": has_kakao,
            "soho_not_isoeun": soho_suspect and no_isoeun_motive,
            "motive_insufficient": not_guilty_yet
            or (soho_suspect and no_isoeun_motive),
        }

    @staticmethod
    def _epitaph_trial2_battle3_logic(user_text: str, selected_evidence_ids: list[str]) -> bool:
        branches = AnswerEvaluatorLLM._epitaph_trial2_battle3_branches(
            user_text, selected_evidence_ids
        )
        return branches["kakao_selected"] and branches["motive_insufficient"]

    @staticmethod
    def epitaph_battle2_is_correct_question(
        text: str,
        selected_evidence_ids: list[str] | None = None,
    ) -> bool:
        """Battle 2: ask why VX was detected on the right hand (question phase, not objection)."""
        lowered = text.lower()
        has_right_hand = "오른손" in text
        has_vx_detection = "vx" in lowered or "검출" in text
        has_ask = any(t in text for t in ("왜", "이유", "경위", "설명", "어떻게"))
        text_match = has_right_hand and has_vx_detection and has_ask

        evidence_ids = set(selected_evidence_ids or [])
        has_medical = "ev_ep_medical" in evidence_ids
        evidence_match = has_medical and has_ask and (has_right_hand or has_vx_detection or "진료" in text)

        return text_match or evidence_match

    def _logic_gate_success(
        self,
        user_text: str,
        statement_id: str,
        selected_evidence_ids: list[str],
    ) -> tuple[bool, str]:
        if statement_id == "stmt_epitaph_isoeun_1" and self._epitaph_battle1_logic(user_text):
            return (
                True,
                "좋습니다. 증언 #1과 VX 용량 정보를 연결해 검찰의 단정 논리를 정확히 흔들었습니다.",
            )
        if statement_id == "stmt_epitaph_isoeun_2" and self._epitaph_battle3_logic(
            user_text, selected_evidence_ids
        ):
            return (
                True,
                "좋습니다. 증언 #1과 #2의 쓰러짐 순서 모순을 정확히 지적했습니다.",
            )
        if statement_id == "stmt_epitaph_isoeun_4" and self._epitaph_battle4_logic(
            user_text, selected_evidence_ids
        ):
            return (
                True,
                "좋습니다. VX 정보와 증언 #4를 연결해 피부 치사량·춤 행동 모순을 지적했습니다.",
            )
        if statement_id == "stmt_minsoo_hack_left" and self._epitaph_trial2_battle1_logic(
            user_text, selected_evidence_ids
        ):
            return (
                True,
                "좋습니다. CCTV 우회전과 서버 로그 좌회전 모순을 정확히 지적했습니다.",
            )
        if statement_id == "counter_minsoo_lidar" and self._epitaph_trial2_battle2_logic(
            user_text, selected_evidence_ids
        ):
            return (
                True,
                "좋습니다. LiDAR 소견 희박성과 살의 시 우회전 코드 논리를 연결했습니다.",
            )
        if statement_id == "counter_kakao_motive" and self._epitaph_trial2_battle3_logic(
            user_text, selected_evidence_ids
        ):
            return (
                True,
                "좋습니다. 카카오가 소호 의심만 보여주며 이소은 살해 동기를 직접 증명하지 못함을 지적했습니다.",
            )
        return False, ""

    def _compute_stage_logic_score(
        self,
        user_text: str,
        current_statement: WitnessTestimonyNode | WitnessCounterStatement,
    ) -> tuple[float, list[str]]:
        text = user_text.lower()
        keyword_bank = {
            "weak_time_of_death": ["부검", "사망", "4시", "5시", "2시", "시체", "발견", "모순"],
            "weak_outage_sound": ["정전", "tv", "비디오", "전기", "작동", "알람", "소리", "모순"],
            "weak_clock_shape": ["장식품", "생각하는", "사람", "시계", "겉보기", "외형", "알람", "소리", "모순"],
            "weak_passport_timezone": ["여권", "뉴욕", "귀국", "시차", "2시간", "늦어", "시간", "모순"],
            "weak_ep_battle1_drink_lethal": ["vx", "20mg", "50mg", "치사량", "용량", "술잔", "단정", "성급", "모순"],
            "weak_ep_battle3_order_contradiction": ["먼저", "순서", "모순", "기절", "정신", "쓰러", "#1", "#2"],
            "weak_ep_battle4_skin_lethal": ["10mg", "피부", "치사량", "춤", "손", "접촉", "모순", "불가능", "독", "#4", "즉사", "쏟"],
            "weak_turn_cctv": ["cctv", "우회전", "좌회전", "서버", "로그", "모순", "방향", "충돌"],
            "weak_lidar_rare": ["희박", "라이다", "소견", "우회전", "좌회전", "살의", "로그", "모순"],
            "weak_motive_not_guilt": ["카카오", "소호", "바텐더", "보복", "이소은", "동기", "무죄", "단정", "의심"],
        }
        keywords = keyword_bank.get(current_statement.weakness_id, [])
        point_keywords: list[str] = []
        for point in current_statement.required_logic_points:
            for token in point.replace(".", " ").replace(",", " ").split():
                if len(token) >= 2:
                    point_keywords.append(token.lower())
        all_keywords = list(dict.fromkeys([k.lower() for k in keywords] + point_keywords[:12]))
        matched_points = [
            point
            for point in current_statement.required_logic_points
            if any(
                token in text
                for token in point.lower().replace(".", " ").split()
                if len(token) >= 2
            )
        ]
        keyword_hits = sum(1 for keyword in all_keywords if keyword and keyword in text)
        logic_score = min(1.0, keyword_hits / max(4, min(len(all_keywords), 8)))
        return logic_score, matched_points

    def _postprocess_stage_evaluation(
        self,
        *,
        current_statement: WitnessTestimonyNode | WitnessCounterStatement,
        user_text: str,
        selected_evidence_ids: list[str],
        evaluation: DefenseArgumentEvaluation,
    ) -> DefenseArgumentEvaluation:
        """Enforce logic-first gates; evidence boosts confidence but is not always required."""
        statement_id = getattr(current_statement, "statement_id", "")
        logic_success, success_reason = self._logic_gate_success(
            user_text, statement_id, selected_evidence_ids
        )
        if logic_success:
            return evaluation.model_copy(
                update={
                    "verdict": AnswerVerdict.SUCCESS,
                    "relevance": RelevanceLevel.RELEVANT,
                    "reason": success_reason,
                }
            )

        if statement_id in self.EPITAPH_SCORING_STATEMENT_IDS:
            focus = self._stage_focus_hint(current_statement)
            return evaluation.model_copy(
                update={
                    "verdict": AnswerVerdict.FAIL,
                    "relevance": RelevanceLevel.PARTIALLY_RELEVANT
                    if user_text.strip()
                    else RelevanceLevel.IRRELEVANT,
                    "reason": (
                        f"핵심 논리 갈래가 부족합니다. {focus}"
                    ),
                }
            )

        required_evidence = set(current_statement.required_evidence_ids or [])
        selected = set(selected_evidence_ids)
        missing = sorted(required_evidence - selected)
        logic_score, matched_points = self._compute_stage_logic_score(user_text, current_statement)
        has_strong_logic = logic_score >= self.STRONG_LOGIC_THRESHOLD or len(matched_points) >= 2

        if missing and not has_strong_logic:
            focus = self._stage_focus_hint(current_statement)
            reason = (
                f"현재 발언의 핵심 모순을 더 구체적으로 짚어 주세요. "
                f"({focus})"
            )
            return evaluation.model_copy(
                update={
                    "verdict": AnswerVerdict.FAIL,
                    "relevance": RelevanceLevel.PARTIALLY_RELEVANT,
                    "reason": reason,
                    "evidence_usage_score": (
                        len(required_evidence & selected) / len(required_evidence)
                        if required_evidence
                        else evaluation.evidence_usage_score
                    ),
                }
            )

        if statement_id == "stmt_epitaph_isoeun_2" and any(
            token in user_text for token in ("오른손", "경위", "설명")
        ):
            reason = (
                "이 질문은 신문 단계에서 유효하지만, 모순 인정에는 "
                "증언 #1과 #2의 순서 충돌을 논리적으로 입증해야 합니다."
            )
            return evaluation.model_copy(
                update={
                    "reason": reason,
                }
            )
        return evaluation

    @staticmethod
    def _stage_focus_hint(current_statement: WitnessTestimonyNode | WitnessCounterStatement) -> str:
        statement_id = getattr(current_statement, "statement_id", "")
        if statement_id == "stmt_epitaph_isoeun_1":
            return "VX 용량(20mg/50mg)과 증언 #1의 독살 단정 논리를 연결해 약점을 찔러야 합니다."
        if statement_id == "stmt_epitaph_isoeun_2":
            return "증언 #1과 #2의 '누가 먼저 쓰러졌는지' 순서 모순을 지적해야 합니다."
        if statement_id == "stmt_epitaph_isoeun_4":
            return "VX 피부 치사량 10mg과 증언 #4의 '독 묻은 손으로 춤' 불가능성을 지적해야 합니다."
        if statement_id == "stmt_minsoo_hack_left":
            return "CCTV 좌회전과 서버 로그 우회전 모순을 함께 제시해야 합니다."
        if statement_id == "counter_minsoo_lidar":
            return "LiDAR 소견 희박성과 살의 시 좌회전 코드·로그 우회전 모순을 지적해야 합니다."
        return "현재 증언의 required_logic_points를 직접 겨냥해 모순을 설명해 주세요."

    def _mock_evaluate(
        self,
        user_answer: str,
        selected_evidence_ids: list[str],
        current_round: TrialRound,
        selected_claim: ProsecutionClaim,
    ) -> AnswerEvaluationResult:
        text = user_answer.lower()
        expected = [p.lower() for p in current_round.expected_defense_points]
        matched = [p for p in expected if p in text]
        related_ev = set(current_round.related_evidence_ids)
        used_related = bool(related_ev & set(selected_evidence_ids))
        core = current_round.core_contradictions[0] if current_round.core_contradictions else None
        weakness_ids = [core.contradiction_id] if core else []

        point_hits = 0
        for point in core.required_points if core else []:
            tokens = [t.lower() for t in point.replace(".", " ").replace(",", " ").split() if len(t) >= 2]
            if any(token in text for token in tokens[:6]):
                point_hits += 1
        core_match = min(1.0, len(matched) * 0.2 + point_hits * 0.15 + (0.25 if used_related else 0))

        if not text.strip():
            verdict = AnswerVerdict.IRRELEVANT
            relevance = RelevanceLevel.IRRELEVANT
        elif core_match >= 0.75 or (core_match >= 0.55 and used_related):
            verdict = AnswerVerdict.SUCCESS
            relevance = RelevanceLevel.RELEVANT
        elif core_match >= 0.5:
            verdict = AnswerVerdict.PARTIAL_SUCCESS
            relevance = RelevanceLevel.PARTIALLY_RELEVANT
        else:
            verdict = AnswerVerdict.FAIL
            relevance = RelevanceLevel.PARTIALLY_RELEVANT

        return AnswerEvaluationResult(
            relevance=relevance,
            core_match_score=core_match,
            logic_score=min(1.0, core_match + 0.1),
            evidence_usage_score=0.9 if used_related else 0.3,
            matched_points=matched,
            missing_points=[],
            incorrect_points=[],
            attacked_claim_ids=[selected_claim.claim_id],
            matched_weakness_ids=weakness_ids if verdict in (AnswerVerdict.SUCCESS, AnswerVerdict.PARTIAL_SUCCESS) else [],
            verdict=verdict,
            reason="현재 주장/증언의 약점을 기준으로 평가했습니다.",
        )

    def _mock_stage_evaluate(
        self,
        user_text: str,
        selected_evidence_ids: list[str],
        current_statement: WitnessTestimonyNode | WitnessCounterStatement,
    ) -> DefenseArgumentEvaluation:
        required_evidence = set(current_statement.required_evidence_ids)
        selected = set(selected_evidence_ids)
        evidence_hits = required_evidence & selected
        evidence_usage_score = (
            1.0
            if required_evidence and required_evidence.issubset(selected)
            else (len(evidence_hits) / len(required_evidence) if required_evidence else 0.5)
        )

        logic_score, matched_points = self._compute_stage_logic_score(user_text, current_statement)
        statement_id = getattr(current_statement, "statement_id", "")
        logic_gate, _ = self._logic_gate_success(user_text, statement_id, selected_evidence_ids)

        if statement_id in self.EPITAPH_SCORING_STATEMENT_IDS:
            if not user_text.strip():
                verdict = AnswerVerdict.IRRELEVANT
                relevance = RelevanceLevel.IRRELEVANT
            elif logic_gate:
                verdict = AnswerVerdict.SUCCESS
                relevance = RelevanceLevel.RELEVANT
            else:
                verdict = AnswerVerdict.FAIL
                relevance = (
                    RelevanceLevel.IRRELEVANT
                    if logic_score < 0.1
                    else RelevanceLevel.PARTIALLY_RELEVANT
                )
            missing_points = [
                point for point in current_statement.required_logic_points if point not in matched_points
            ]
            return DefenseArgumentEvaluation(
                relevance=relevance,
                core_match_score=1.0 if logic_gate else min(0.4, logic_score),
                logic_score=logic_score,
                evidence_usage_score=evidence_usage_score,
                matched_points=matched_points,
                missing_points=missing_points,
                incorrect_points=[],
                verdict=verdict,
                target_weakness_id=current_statement.weakness_id,
                reason=(
                    "현재 증언의 핵심 논리 갈래를 모두 충족했습니다."
                    if logic_gate
                    else "핵심 논리 갈래가 부족합니다. "
                    + self._stage_focus_hint(current_statement)
                ),
            )

        has_strong_logic = logic_gate or logic_score >= self.STRONG_LOGIC_THRESHOLD or len(matched_points) >= 2
        core_match_score = min(1.0, logic_score * 0.85 + evidence_usage_score * 0.15)

        if not user_text.strip():
            verdict = AnswerVerdict.IRRELEVANT
            relevance = RelevanceLevel.IRRELEVANT
        elif logic_score < 0.15 and not evidence_hits:
            verdict = AnswerVerdict.IRRELEVANT
            relevance = RelevanceLevel.IRRELEVANT
        elif logic_gate or (has_strong_logic and logic_score >= 0.35):
            verdict = AnswerVerdict.SUCCESS
            relevance = RelevanceLevel.RELEVANT
        elif evidence_usage_score >= 0.75 and logic_score >= 0.35:
            verdict = AnswerVerdict.SUCCESS
            relevance = RelevanceLevel.RELEVANT
        elif (evidence_usage_score >= 0.5 and logic_score >= 0.25) or logic_score >= 0.35:
            verdict = AnswerVerdict.PARTIAL_SUCCESS
            relevance = RelevanceLevel.PARTIALLY_RELEVANT
        else:
            verdict = AnswerVerdict.FAIL
            relevance = RelevanceLevel.PARTIALLY_RELEVANT

        missing_points = [
            point for point in current_statement.required_logic_points if point not in matched_points
        ]
        return DefenseArgumentEvaluation(
            relevance=relevance,
            core_match_score=core_match_score,
            logic_score=logic_score,
            evidence_usage_score=evidence_usage_score,
            matched_points=matched_points,
            missing_points=missing_points,
            incorrect_points=[],
            verdict=verdict,
            target_weakness_id=current_statement.weakness_id,
            reason="현재 증언의 논리적 결함과 주장의 일치를 기준으로 평가했습니다.",
        )
