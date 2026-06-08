from typing import Any, Awaitable, Callable

from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM
from backend.ai_services.dialogue_router import orchestrate_player_turn
from backend.ai_services.judge_actor import JudgeActorLLM
from backend.ai_services.prosecutor_actor import ProsecutorActorLLM
from backend.ai_services.witness_actor import WitnessActorLLM
from backend.core.character_knowledge import CharacterKnowledgeManager
from backend.core.helper import Helper
from backend.logging_config import get_logger
from backend.core.response_verifier import ResponseVerifierLLM
from backend.core.state_manager import StateManager
from backend.schemas.court import ActorLine, ActorResponse, CourtRecord, TruthStatus
from backend.schemas.episode import EpisodeData, StageType, TrialStage
from backend.schemas.trial import AnswerVerdict, ContradictionSeverity, DefenseArgumentEvaluation, RelevanceLevel, StageResult, TurnContradictionEvaluation


class FreeDialogueEngine:
    PASSIVE_JUDGE_THRESHOLD = 3
    SUCCESS_JUDGE_EVENT_TYPES = frozenset(
        {"objection_sustained", "argument_success", "partial_success"}
    )
    logger = get_logger(__name__)

    def __init__(
        self,
        state: StateManager,
        evaluator: AnswerEvaluatorLLM,
        witness_actor: WitnessActorLLM,
        prosecutor_actor: ProsecutorActorLLM,
        judge_actor: JudgeActorLLM,
    ):
        self.state = state
        self.evaluator = evaluator
        self.witness_actor = witness_actor
        self.prosecutor_actor = prosecutor_actor
        self.judge_actor = judge_actor
        self.helper = Helper()
        self.knowledge = CharacterKnowledgeManager()
        self.response_verifier = ResponseVerifierLLM()

    async def process(
        self,
        session_id: str,
        episode: EpisodeData,
        stage: TrialStage,
        text: str,
        mode: str,
        selected_evidence_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        selected_evidence_ids = selected_evidence_ids or []
        if len(selected_evidence_ids) > 2:
            return [{"type": "error", "message": "증거는 최대 2개까지 선택할 수 있습니다."}]
        if len(text) > 100:
            return [{"type": "error", "message": "주장은 100자 이내로 입력해야 합니다."}]
        if not text.strip():
            return [{"type": "error", "message": "내용을 입력해 주세요."}]

        selected_evidence_ids = await self._sanitize_selected_evidence_ids(
            session_id, selected_evidence_ids
        )

        return await self.process_player_turn(
            session_id, episode, stage, text, mode, selected_evidence_ids
        )

    async def _sanitize_selected_evidence_ids(
        self,
        session_id: str,
        selected_evidence_ids: list[str],
    ) -> list[str]:
        inventory = set(await self.state.get_inventory(session_id))
        ts = await self.state.get_trial_state(session_id)
        court_records = await self.state.get_court_records(session_id)
        allowed_statements = set(ts.usable_statement_evidence_ids) | {
            record.statement_id for record in court_records
        }
        allowed = inventory | allowed_statements
        return [evidence_id for evidence_id in selected_evidence_ids if evidence_id in allowed]

    async def process_player_turn(
        self,
        session_id: str,
        episode: EpisodeData,
        stage: TrialStage,
        text: str,
        mode: str,
        selected_evidence_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        selected_evidence_ids = selected_evidence_ids or []
        ts = await self.state.get_trial_state(session_id)
        if self._is_epitaph_battle2_pending(ts, stage):
            battle2_events = await self._process_epitaph_battle2_turn(
                session_id,
                episode,
                stage,
                text,
                mode,
                selected_evidence_ids,
            )
            battle2_events.append(
                {
                    "type": "orchestration_complete",
                    "stage_id": stage.stage_id,
                    "mode": mode,
                    "responders": ["battle2_interrogation"],
                }
            )
            return battle2_events

        effective_mode = mode
        if (
            mode == "question"
            and selected_evidence_ids
            and ts.stage_phase == "cross_exam_free"
        ):
            effective_mode = "objection"

        plan = orchestrate_player_turn(
            text,
            stage,
            ts.free_dialogue_exchanges + 1,
            effective_mode,
            threshold=self.PASSIVE_JUDGE_THRESHOLD,
            stage_phase=ts.stage_phase,
        )

        await self.state.append_dialogue(session_id, "player", text)
        ts = await self.state.get_trial_state(session_id)
        ts.free_dialogue_exchanges = ts.free_dialogue_exchanges + 1
        ts.free_dialogue_history = ts.free_dialogue_history + [
            {
                "speaker": "player",
                "text": text,
                "addressee": plan.primary_addressee,
                "mode": effective_mode,
            }
        ]
        ts.last_addressee = plan.primary_addressee
        await self.state.save_trial_state(session_id, ts)

        events: list[dict[str, Any]] = [
            {
                "type": "orchestration_planned",
                "stage_id": stage.stage_id,
                "mode": effective_mode,
                "stage_phase": ts.stage_phase,
                "responders": plan.responders,
                "primary_addressee": plan.primary_addressee,
                "judge_trigger": plan.judge_trigger,
                "reason": plan.reason,
            },
            {
                "type": "addressee_routed",
                "addressee": plan.primary_addressee,
                "stage_id": stage.stage_id,
            },
        ]

        turn_batch_lines: list[dict[str, str]] = []

        for batch_position, responder in enumerate(plan.responders):
            context_addressee = self._responder_addressee(responder, stage, plan.primary_addressee)
            context = await self._build_free_dialogue_context(
                session_id,
                episode,
                stage,
                context_addressee,
                await self.state.get_trial_state(session_id),
            )
            context["turn_batch_lines"] = turn_batch_lines
            context["batch_position"] = batch_position

            if responder == "witness":
                witness_id = stage.active_witness_id or plan.primary_addressee
                context["response_mode"] = "answer_player"
                context["max_response_lines"] = 2 if ts.stage_phase == "cross_exam_free" else 1
                response = await self.witness_actor.respond_to_free_question(
                    user_question=text,
                    witness_id=witness_id,
                    current_statement=context["stage_context"]["current_statement"],
                    witness_mental=context["witness_mental"],
                    dialogue_context=context,
                )
                response, verification = await self._verify_actor_response(
                    role="witness",
                    stage=stage,
                    stage_phase=ts.stage_phase,
                    user_text=text,
                    response=response,
                    current_statement=context["stage_context"]["current_statement"],
                    turn_batch_lines=turn_batch_lines,
                    forbidden_claims=episode.forbidden_claims,
                    inventory_evidence=context.get("inventory_evidence") or [],
                    retry_fn=lambda feedback: self.witness_actor.respond_to_free_question(
                        user_question=text,
                        witness_id=witness_id,
                        current_statement=context["stage_context"]["current_statement"],
                        witness_mental=context["witness_mental"],
                        dialogue_context=context,
                        verifier_feedback=feedback,
                    ),
                    fallback_speaker=witness_id,
                )
                await self._persist_actor_lines(session_id, response.lines)
                turn_batch_lines.extend(
                    self._lines_to_batch_entries(response.lines, witness_id)
                )
                events.append(
                    {
                        "type": "witness_reaction",
                        "stage_id": stage.stage_id,
                        "addressee": witness_id,
                        "batch_position": batch_position,
                        "expression_state": self._line_expression(response.lines, "basic"),
                        "animation_tag": self._line_expression(response.lines, "basic"),
                        "lines": [ln.model_dump() for ln in response.lines],
                        "verification": verification,
                    }
                )
            elif responder == "witness_followup":
                witness_id = stage.active_witness_id or plan.primary_addressee
                context["response_mode"] = "followup_after_prosecutor"
                context["max_response_lines"] = 1
                response = await self.witness_actor.respond_to_free_question(
                    user_question=text,
                    witness_id=witness_id,
                    current_statement=context["stage_context"]["current_statement"],
                    witness_mental=context["witness_mental"],
                    dialogue_context=context,
                )
                response, verification = await self._verify_actor_response(
                    role="witness",
                    stage=stage,
                    stage_phase=ts.stage_phase,
                    user_text=text,
                    response=response,
                    current_statement=context["stage_context"]["current_statement"],
                    turn_batch_lines=turn_batch_lines,
                    forbidden_claims=episode.forbidden_claims,
                    inventory_evidence=context.get("inventory_evidence") or [],
                    retry_fn=lambda feedback: self.witness_actor.respond_to_free_question(
                        user_question=text,
                        witness_id=witness_id,
                        current_statement=context["stage_context"]["current_statement"],
                        witness_mental=context["witness_mental"],
                        dialogue_context=context,
                        verifier_feedback=feedback,
                    ),
                    fallback_speaker=witness_id,
                )
                await self._persist_actor_lines(session_id, response.lines)
                turn_batch_lines.extend(
                    self._lines_to_batch_entries(response.lines, witness_id)
                )
                events.append(
                    {
                        "type": "witness_reaction",
                        "stage_id": stage.stage_id,
                        "addressee": witness_id,
                        "batch_position": batch_position,
                        "source": "witness_followup",
                        "expression_state": self._line_expression(response.lines, "basic"),
                        "animation_tag": self._line_expression(response.lines, "basic"),
                        "lines": [ln.model_dump() for ln in response.lines],
                        "verification": verification,
                    }
                )
            elif responder == "prosecutor":
                context["response_mode"] = (
                    "defend_witness"
                    if plan.primary_addressee != "pros_001"
                    else "counter_defense"
                )
                context["max_response_lines"] = (
                    2 if context["response_mode"] == "counter_defense" else 2
                )
                response = await self.prosecutor_actor.respond_to_free_question(
                    user_question=text,
                    stage=stage,
                    episode=episode,
                    dialogue_context=context,
                )
                response, verification = await self._verify_actor_response(
                    role="prosecutor",
                    stage=stage,
                    stage_phase=ts.stage_phase,
                    user_text=text,
                    response=response,
                    current_statement=context["stage_context"]["current_statement"],
                    turn_batch_lines=turn_batch_lines,
                    forbidden_claims=episode.forbidden_claims,
                    inventory_evidence=context.get("inventory_evidence") or [],
                    retry_fn=lambda feedback: self.prosecutor_actor.respond_to_free_question(
                        user_question=text,
                        stage=stage,
                        episode=episode,
                        dialogue_context=context,
                        verifier_feedback=feedback,
                    ),
                    fallback_speaker="pros_001",
                )
                await self._persist_actor_lines(session_id, response.lines)
                turn_batch_lines.extend(
                    self._lines_to_batch_entries(response.lines, "pros_001")
                )
                events.append(
                    {
                        "type": "prosecutor_response",
                        "stage_id": stage.stage_id,
                        "addressee": "pros_001",
                        "batch_position": batch_position,
                        "response_mode": context["response_mode"],
                        "mode": context["response_mode"],
                        "expression_state": self._prosecutor_expression(context["response_mode"], response.lines),
                        "animation_tag": self._prosecutor_expression(context["response_mode"], response.lines),
                        "lines": [ln.model_dump() for ln in response.lines],
                        "verification": verification,
                    }
                )
            elif responder == "judge":
                judge_events = await self._build_judge_response(
                    session_id,
                    episode,
                    stage,
                    text,
                    selected_evidence_ids,
                    plan.judge_trigger,
                    effective_mode,
                )
                events.extend(judge_events)
                evaluation_event = next(
                    (ev for ev in judge_events if ev.get("type") == "defense_argument_evaluated"),
                    None,
                )
                if evaluation_event:
                    evaluation = DefenseArgumentEvaluation.model_validate(
                        evaluation_event["evaluation"]
                    )
                    if self._should_emit_success_cheer(evaluation, judge_events):
                        current_stmt = self._current_statement(stage, ts)
                        broken_index = (
                            self.helper.get_statement_chain_index(
                                stage, current_stmt.statement_id
                            )
                            if current_stmt and hasattr(current_stmt, "statement_id")
                            else None
                        )
                        will_emit_contradiction_helper = (
                            ts.helper_enabled
                            and broken_index is not None
                            and self._next_counter_id(current_stmt) is not None
                            and self.helper.get_helper_lines_after_break(stage, broken_index)
                        )
                        if not will_emit_contradiction_helper:
                            events.append(self._success_cheer_event(stage.stage_id))
                    if effective_mode == "objection" and plan.judge_trigger == "objection":
                        transition_events = await self._apply_objection_outcome(
                            session_id,
                            episode,
                            stage,
                            evaluation,
                        )
                        events.extend(transition_events)
                    elif evaluation.verdict == AnswerVerdict.SUCCESS:
                        success_events = await self._handle_successful_evaluation(
                            session_id, episode, stage, evaluation
                        )
                        events.extend(success_events)

        if self._should_evaluate_turn_contradiction(plan, effective_mode, turn_batch_lines, ts.stage_phase) and not self._should_skip_turn_contradiction(
            stage=stage,
            ts=ts,
            mode=effective_mode,
            selected_evidence_ids=selected_evidence_ids,
            text=text,
        ):
            turn_events = await self._evaluate_turn_contradiction(
                session_id,
                episode,
                stage,
                text,
                effective_mode,
                turn_batch_lines,
                selected_evidence_ids,
                judge_trigger=plan.judge_trigger,
            )
            events.extend(turn_events)

        events.append(
            {
                "type": "orchestration_complete",
                "stage_id": stage.stage_id,
                "mode": effective_mode,
                "responders": plan.responders,
            }
        )
        return events

    @staticmethod
    def _should_evaluate_turn_contradiction(
        plan,
        mode: str,
        turn_batch_lines: list[dict[str, str]],
        stage_phase: str,
    ) -> bool:
        if mode == "objection":
            return False
        if "judge" in plan.responders:
            return False
        if not turn_batch_lines:
            return False
        return stage_phase == "cross_exam_free"

    @staticmethod
    def _turn_evaluation_favors_defense(evaluation: TurnContradictionEvaluation) -> bool:
        if evaluation.intervention_needed:
            return False
        if evaluation.severity == ContradictionSeverity.SEVERE:
            return False
        return evaluation.verdict in (AnswerVerdict.SUCCESS, AnswerVerdict.PARTIAL_SUCCESS)

    def _should_skip_turn_contradiction(
        self,
        *,
        stage: TrialStage,
        ts: Any,
        mode: str,
        selected_evidence_ids: list[str],
        text: str,
    ) -> bool:
        if stage.stage_id != "stage_epitaph_club":
            return False
        if mode != "question":
            return False
        if selected_evidence_ids:
            return False
        current_statement = self._current_statement(stage, ts)
        if not current_statement:
            return False
        if getattr(current_statement, "statement_id", "") != "stmt_epitaph_isoeun_2":
            return False
        # Battle 2 is a pure free-question phase (right-hand explanation), not a scoring contradiction turn.
        return any(token in text for token in ("오른손", "손", "경위", "설명", "왜"))

    @staticmethod
    def _is_epitaph_battle2_pending(ts: Any, stage: TrialStage) -> bool:
        if stage.stage_id != "stage_epitaph_club":
            return False
        if ts.stage_phase != "cross_exam_free":
            return False
        return "stmt_epitaph_isoeun_2" not in ts.usable_statement_evidence_ids

    @staticmethod
    def _is_epitaph_battle2_correct_question(
        text: str,
        selected_evidence_ids: list[str] | None = None,
    ) -> bool:
        return AnswerEvaluatorLLM.epitaph_battle2_is_correct_question(text, selected_evidence_ids)

    async def _emit_epitaph_trial1_stage_clear_script(
        self,
        session_id: str,
        stage: TrialStage,
    ) -> list[dict[str, Any]]:
        """Fixed confession → prosecutor adjourn request → judge adjourn (no LLM)."""
        ctx = stage.prosecution_context or {}
        witness_id = stage.active_witness_id or "wit_ep_001"
        confession = str(ctx.get("fixed_confession_line") or "").strip()
        prosecutor_text = str(ctx.get("fixed_prosecutor_adjourn_line") or "").strip()
        judge_text = str(ctx.get("fixed_judge_adjourn_line") or "").strip()

        actor_lines: list[ActorLine] = []
        events: list[dict[str, Any]] = []

        if confession:
            breakdown_line = ActorLine(
                speaker=witness_id,
                dialogue=confession,
                animation_tag="breakdown",
            )
            actor_lines.append(breakdown_line)
            events.append(
                {
                    "type": "witness_breakdown",
                    "stage_id": stage.stage_id,
                    "witness_id": witness_id,
                    "witness_mental_band": "breakdown",
                    "expression_state": "breakdown",
                    "is_fixed": True,
                    "lines": [breakdown_line.model_dump()],
                }
            )

        if prosecutor_text:
            prosecutor_line = ActorLine(
                speaker="pros_001",
                dialogue=prosecutor_text,
                animation_tag="think",
            )
            actor_lines.append(prosecutor_line)
            events.append(
                {
                    "type": "prosecutor_pressure",
                    "stage_id": stage.stage_id,
                    "intervention_type": "trial_adjourn_request",
                    "is_fixed": True,
                    "lines": [prosecutor_line.model_dump()],
                }
            )

        if judge_text:
            judge_line = ActorLine(
                speaker="judge_001",
                dialogue=judge_text,
                animation_tag="success",
            )
            actor_lines.append(judge_line)
            events.append(
                {
                    "type": "judge_comment",
                    "stage_id": stage.stage_id,
                    "event_type": "trial_adjourned",
                    "is_fixed": True,
                    "lines": [judge_line.model_dump()],
                    "sfx": "sfx_gavel_3",
                }
            )

        if actor_lines:
            await self._persist_actor_lines(session_id, actor_lines)
        return events

    async def _emit_epitaph_battle2_intro(
        self,
        session_id: str,
        stage: TrialStage,
    ) -> list[dict[str, Any]]:
        ctx = stage.prosecution_context or {}
        judge_text = ctx.get("fixed_judge_cross_exam_prompt") or "변호인 할 말 있습니까?"
        defense_text = ctx.get("fixed_defense_cross_exam_intent") or "증인에게 물어보고 싶은 것이 있습니다."
        judge_line = ActorLine(speaker="judge_001", dialogue=judge_text, animation_tag="think")
        defense_line = ActorLine(speaker="player", dialogue=defense_text, animation_tag="basic")
        await self._persist_actor_lines(session_id, [judge_line, defense_line])
        return [
            {
                "type": "actor_lines",
                "stage_id": stage.stage_id,
                "is_fixed": True,
                "lines": [judge_line.model_dump(), defense_line.model_dump()],
            }
        ]

    async def _process_epitaph_battle2_turn(
        self,
        session_id: str,
        episode: EpisodeData,
        stage: TrialStage,
        text: str,
        mode: str,
        selected_evidence_ids: list[str],
    ) -> list[dict[str, Any]]:
        if mode != "question":
            return [
                {
                    "type": "error",
                    "message": "지금은 증인에게 질문해야 합니다. (전송으로 질문하세요)",
                }
            ]

        events: list[dict[str, Any]] = [
            {
                "type": "orchestration_planned",
                "stage_id": stage.stage_id,
                "mode": mode,
                "stage_phase": "cross_exam_free",
                "responders": ["battle2_interrogation"],
                "primary_addressee": stage.active_witness_id,
                "judge_trigger": "none",
                "reason": "epitaph_battle2_interrogation",
            }
        ]

        ts = await self.state.get_trial_state(session_id)
        current_statement = self._current_statement(stage, ts)
        if not current_statement:
            return events + [{"type": "error", "message": "현재 증언 상태를 찾을 수 없습니다."}]

        if self._is_epitaph_battle2_correct_question(text, selected_evidence_ids):
            counter_events = await self._emit_fixed_counter_statement(
                session_id, stage, current_statement
            )
            events.extend(counter_events)
            events.append(
                {
                    "type": "battle2_interrogation_resolved",
                    "stage_id": stage.stage_id,
                    "success": True,
                }
            )
            return events

        life_loss = self._default_life_loss(stage, ts)
        if life_loss > 0:
            events.extend(
                await self._append_life_penalty_events(session_id, stage.stage_id, life_loss)
            )

        witness_id = stage.active_witness_id or "wit_ep_001"
        judge_line = ActorLine(
            speaker="judge_001",
            dialogue="질문의 초점을 현재 쟁점에 맞추십시오.",
            animation_tag="serious",
        )
        prosecutor_line = ActorLine(
            speaker="pros_001",
            dialogue="변호인은 지금 사건과 무관한 질문을 하고 있습니다!",
            animation_tag="basic",
        )
        witness_line = ActorLine(
            speaker=witness_id,
            dialogue="……",
            animation_tag="embarrassed",
        )
        await self._persist_actor_lines(
            session_id, [judge_line, prosecutor_line, witness_line]
        )
        events.extend(
            [
                {
                    "type": "judge_intervention",
                    "event_type": "argument_fail",
                    "trigger": "battle2_wrong_question",
                    "severity": "moderate",
                    "expression_state": "serious",
                    "animation_tag": "serious",
                    "lines": [judge_line.model_dump()],
                    "judge_comment": judge_line.dialogue,
                    "is_fixed": True,
                    "sfx": "sfx_gavel_1",
                },
                {
                    "type": "prosecutor_response",
                    "stage_id": stage.stage_id,
                    "addressee": "pros_001",
                    "response_mode": "objection",
                    "mode": "objection",
                    "lines": [prosecutor_line.model_dump()],
                    "is_fixed": True,
                },
                {
                    "type": "witness_reaction",
                    "stage_id": stage.stage_id,
                    "witness_id": witness_id,
                    "addressee": witness_id,
                    "source": "battle2_wrong_question",
                    "lines": [witness_line.model_dump()],
                    "is_fixed": True,
                },
            ]
        )
        events.append(
            {
                "type": "battle2_interrogation_resolved",
                "stage_id": stage.stage_id,
                "success": False,
            }
        )
        return events

    async def _evaluate_turn_contradiction(
        self,
        session_id: str,
        episode: EpisodeData,
        stage: TrialStage,
        text: str,
        mode: str,
        turn_batch_lines: list[dict[str, str]],
        selected_evidence_ids: list[str],
        judge_trigger: str = "none",
    ) -> list[dict[str, Any]]:
        ts = await self.state.get_trial_state(session_id)
        current_statement = self._current_statement(stage, ts)
        current_statement_dict = (
            current_statement.model_dump()
            if current_statement and hasattr(current_statement, "model_dump")
            else None
        )
        witness_mental = ts.witness_mental_by_stage.get(stage.stage_id, stage.witness_mental)
        judge_persuasion = ts.judge_persuasion_by_stage.get(stage.stage_id, stage.judge_persuasion)

        evaluation, judge_response = await self.judge_actor.evaluate_turn_contradiction(
            stage_type=stage.stage_type.value,
            stage_phase=ts.stage_phase,
            mode=mode,
            user_text=text,
            turn_batch_lines=turn_batch_lines,
            current_statement=current_statement_dict,
            remaining_life=ts.stage_life,
            witness_mental=witness_mental,
            judge_persuasion=judge_persuasion,
        )

        events: list[dict[str, Any]] = [
            {
                "type": "turn_contradiction_evaluated",
                "stage_id": stage.stage_id,
                "evaluation": evaluation.model_dump(),
                "turn_batch_lines": turn_batch_lines,
                "player_text": text,
            }
        ]

        defense_favorable = self._turn_evaluation_favors_defense(evaluation)
        if not evaluation.intervention_needed and not defense_favorable:
            return events

        if defense_favorable and not judge_response.lines:
            judge_response = self.judge_actor.build_turn_sustain_response(
                evaluation, evaluation.reason
            )
        if not judge_response.lines:
            return events

        await self._persist_actor_lines(session_id, judge_response.lines)

        trigger = judge_trigger if judge_trigger != "none" else "rebuttal_contradiction"
        if defense_favorable and not evaluation.intervention_needed:
            event_type = (
                "objection_sustained"
                if evaluation.verdict == AnswerVerdict.SUCCESS
                else "partial_success"
            )
        elif evaluation.severity == ContradictionSeverity.SEVERE:
            event_type = "argument_fail"
        else:
            event_type = "partial_success"

        defense_eval = DefenseArgumentEvaluation(
            relevance=RelevanceLevel.PARTIALLY_RELEVANT
            if evaluation.verdict == AnswerVerdict.PARTIAL_SUCCESS
            else RelevanceLevel.IRRELEVANT
            if evaluation.verdict == AnswerVerdict.IRRELEVANT
            else RelevanceLevel.RELEVANT,
            core_match_score=0.2 if evaluation.verdict == AnswerVerdict.FAIL else 0.5,
            logic_score=0.2 if evaluation.verdict == AnswerVerdict.FAIL else 0.5,
            evidence_usage_score=0.3 if selected_evidence_ids else 0.0,
            verdict=evaluation.verdict,
            reason=evaluation.reason,
        )

        expression = self._line_expression(
            judge_response.lines,
            "success" if defense_favorable else "serious",
        )
        events.extend(
            [
                {
                    "type": "defense_argument_evaluated",
                    "stage_id": stage.stage_id,
                    "evaluation": defense_eval.model_dump(),
                    "mode": "free_dialogue",
                    "selected_evidence_ids": selected_evidence_ids,
                    "judge_comment": evaluation.reason,
                    "trigger": trigger,
                },
                {
                    "type": "judge_intervention",
                    "event_type": event_type,
                    "trigger": trigger,
                    "severity": evaluation.severity.value,
                    "expression_state": expression,
                    "animation_tag": expression,
                    "lines": [ln.model_dump() for ln in judge_response.lines],
                    "judge_comment": evaluation.reason,
                    "sfx": "sfx_gavel_1",
                },
            ]
        )

        if event_type in ("partial_success", "objection_sustained"):
            events.append(self._success_cheer_event(stage.stage_id))

        life_loss = 0
        if evaluation.intervention_needed and not defense_favorable:
            life_loss = evaluation.life_loss
            if life_loss <= 0:
                ts = await self.state.get_trial_state(session_id)
                if ts.stage_life > 0:
                    life_loss = self._default_life_loss(stage, ts)

        if life_loss > 0:
            events.extend(
                await self._append_life_penalty_events(session_id, stage.stage_id, life_loss)
            )

        if evaluation.persuasion_delta > 0:
            persuasion = await self.state.apply_judge_persuasion(
                session_id, stage.stage_id, evaluation.persuasion_delta
            )
            events.append(
                {
                    "type": "judge_persuasion_update",
                    "stage_id": stage.stage_id,
                    "judge_persuasion": persuasion,
                    "judge_persuasion_band": "high" if persuasion >= 70 else "medium",
                }
            )

        return events

    async def _apply_objection_outcome(
        self,
        session_id: str,
        episode: EpisodeData,
        stage: TrialStage,
        evaluation: DefenseArgumentEvaluation,
    ) -> list[dict[str, Any]]:
        if evaluation.verdict != AnswerVerdict.SUCCESS:
            return []

        ts = await self.state.get_trial_state(session_id)
        was_testimony_phase = ts.stage_phase == "testimony"

        events = await self._handle_successful_evaluation(session_id, episode, stage, evaluation)
        if any(ev.get("type") == "stage_cleared" for ev in events):
            return events

        if was_testimony_phase:
            ts = await self.state.get_trial_state(session_id)
            ts.stage_phase = "cross_exam_free"
            await self.state.save_trial_state(session_id, ts)
            await self.state.reset_stage_hint_level(
                session_id, stage.stage_id, phase="cross_exam_free"
            )
            if stage.stage_id == "stage_epitaph_club":
                events.extend(await self._emit_epitaph_battle2_intro(session_id, stage))
            events.append(
                {
                    "type": "phase_transition",
                    "stage_id": stage.stage_id,
                    "from_phase": "testimony",
                    "to_phase": "cross_exam_free",
                    "message": "증인에게 물어보고 싶은 것이 있습니다."
                    if stage.stage_id == "stage_epitaph_club"
                    else "증인의 증언에 모순이 드러났습니다.",
                    "is_fixed": stage.stage_id == "stage_epitaph_club",
                    "battle2_interrogation": stage.stage_id == "stage_epitaph_club",
                    "sfx": "sfx_gavel_1",
                }
            )

        return events

    async def _handle_successful_evaluation(
        self,
        session_id: str,
        episode: EpisodeData,
        stage: TrialStage,
        evaluation: DefenseArgumentEvaluation,
    ) -> list[dict[str, Any]]:
        """Apply mental damage and emit the next scripted counter from the episode chain."""
        ts = await self.state.get_trial_state(session_id)
        current_statement = self._current_statement(stage, ts)
        if not current_statement or not hasattr(current_statement, "statement_id"):
            return []

        events: list[dict[str, Any]] = []
        damage = getattr(current_statement, "damage_on_success", 25)
        mental = await self.state.apply_witness_mental_damage(session_id, stage.stage_id, damage)
        await self.state.mark_statement_weakened(session_id, current_statement.statement_id)
        if current_statement.statement_id == "stmt_epitaph_isoeun_1":
            events.extend(
                await self._emit_epitaph_doctor_opinion_submission(session_id, stage.stage_id, stage)
            )
        if (
            stage.stage_id == "stage_epitaph_car"
            and current_statement.statement_id == "stmt_minsoo_hack_left"
        ):
            events.extend(await self._emit_epitaph_trial2_battle2_intro(session_id, stage))
        if (
            stage.stage_id == "stage_epitaph_car"
            and current_statement.statement_id == "counter_minsoo_lidar"
        ):
            events.extend(await self._emit_epitaph_trial2_battle3_intro(session_id, stage))
        events.append(
            {
                "type": "witness_mental_update",
                "stage_id": stage.stage_id,
                "witness_id": stage.active_witness_id,
                "remaining_witness_mental": mental,
                "witness_mental_band": self._mental_band(mental),
                "mental_damage": damage,
                "expression_state": self._witness_expression(mental),
                "animation_tag": self._witness_expression(mental),
                "camera_effect": "flash",
                "crowd_reaction": "murmur",
            }
        )

        clear_threshold = stage.clear_condition.witness_mental_lte if stage.clear_condition else None
        if clear_threshold is not None and mental <= clear_threshold:
            result = await self._clear_stage(session_id, stage)
            if stage.stage_id == "stage_epitaph_club":
                events.extend(await self._emit_epitaph_trial1_stage_clear_script(session_id, stage))
            if stage.stage_id == "stage_epitaph_car":
                events.extend(await self._emit_epitaph_trial2_stage_clear_script(session_id, stage))
            events.append(
                {
                    "type": "stage_cleared",
                    **result.model_dump(),
                    "sfx": "sfx_gavel_3",
                    "animation_tag": "success",
                    "camera_effect": "flash",
                    "crowd_reaction": "murmur",
                }
            )
            return events

        # Battle 2 (right-hand interrogation) unlocks stmt #2 only after a valid player question.
        if (
            stage.stage_id == "stage_epitaph_club"
            and current_statement.statement_id == "stmt_epitaph_isoeun_1"
        ):
            return events

        counter_events = await self._emit_fixed_counter_statement(session_id, stage, current_statement)
        events.extend(counter_events)

        return events

    @classmethod
    def _should_emit_success_cheer(
        cls,
        evaluation: DefenseArgumentEvaluation,
        judge_events: list[dict[str, Any]],
    ) -> bool:
        if evaluation.verdict == AnswerVerdict.SUCCESS:
            return True
        intervention = next(
            (ev for ev in judge_events if ev.get("type") == "judge_intervention"),
            None,
        )
        if not intervention:
            return False
        return intervention.get("event_type") in cls.SUCCESS_JUDGE_EVENT_TYPES

    def _success_cheer_event(self, stage_id: str) -> dict[str, Any]:
        return {
            "type": "helper_success_cheer",
            "stage_id": stage_id,
            "helper_lines": self.helper.get_success_cheer_lines(),
        }

    async def _emit_fixed_counter_statement(
        self,
        session_id: str,
        stage: TrialStage,
        current_statement: Any,
    ) -> list[dict[str, Any]]:
        counter_id = self._next_counter_id(current_statement)
        if not counter_id:
            return []

        counter = stage.counter_by_id(counter_id)
        if not counter:
            return []

        witness_id = stage.active_witness_id or "wit_001"
        await self._add_statement_record(
            session_id,
            stage,
            counter.statement_id,
            counter.text,
            "witness_counter",
            usable=True,
        )
        broken_index = self.helper.get_statement_chain_index(
            stage, current_statement.statement_id
        )
        if broken_index is None:
            broken_index = 0

        await self.state.set_current_statement(
            session_id, testimony_id=None, counter_statement_id=counter.statement_id
        )
        ts = await self.state.get_trial_state(session_id)
        await self.state.reset_stage_hint_level(
            session_id, stage.stage_id, phase=ts.stage_phase or "testimony"
        )

        lines = [ActorLine(speaker=witness_id, dialogue=counter.text, animation_tag="embarrassed")]
        await self._persist_actor_lines(session_id, lines)

        events: list[dict[str, Any]] = [
            {
                "type": "usable_statement_added",
                "is_fixed": bool(getattr(counter, "is_fixed", False)),
                "record": {
                    "statement_id": counter.statement_id,
                    "speaker": witness_id,
                    "text": counter.text,
                    "source": "witness_counter",
                    "usable_as_evidence": True,
                    "stage_id": stage.stage_id,
                    "is_fixed": bool(getattr(counter, "is_fixed", False)),
                },
            },
            {
                "type": "witness_reaction",
                "stage_id": stage.stage_id,
                "witness_id": witness_id,
                "addressee": witness_id,
                "statement_id": counter.statement_id,
                "source": "fixed_testimony",
                "is_fixed": bool(getattr(counter, "is_fixed", False)),
                "lines": [ln.model_dump() for ln in lines],
                "sfx": "sfx_gavel_1",
            },
            {"type": "court_record_updated"},
        ]

        if ts.helper_enabled:
            helper_lines = self.helper.get_helper_lines_after_break(stage, broken_index)
            if helper_lines:
                events.append(
                    {
                        "type": "contradiction_helper",
                        "stage_id": stage.stage_id,
                        "helper_lines": helper_lines,
                        "broken_index": broken_index,
                        "next_stage": broken_index + 1,
                    }
                )

        # Epitaph trial 1 scripted bridge:
        # once #3 appears, immediately continue to fixed #4 without another success input.
        if (
            stage.stage_id == "stage_epitaph_club"
            and counter.statement_id == "stmt_epitaph_isoeun_3"
            and counter.next_counter_statement_id
        ):
            next_counter = stage.counter_by_id(counter.next_counter_statement_id)
            if next_counter:
                await self._add_statement_record(
                    session_id,
                    stage,
                    next_counter.statement_id,
                    next_counter.text,
                    "witness_counter",
                    usable=True,
                )
                await self.state.set_current_statement(
                    session_id,
                    testimony_id=None,
                    counter_statement_id=next_counter.statement_id,
                )
                next_lines = [
                    ActorLine(
                        speaker=witness_id,
                        dialogue=next_counter.text,
                        animation_tag="sweat",
                    )
                ]
                await self._persist_actor_lines(session_id, next_lines)
                events.extend(
                    [
                        {
                            "type": "usable_statement_added",
                            "is_fixed": bool(getattr(next_counter, "is_fixed", False)),
                            "record": {
                                "statement_id": next_counter.statement_id,
                                "speaker": witness_id,
                                "text": next_counter.text,
                                "source": "witness_counter",
                                "usable_as_evidence": True,
                                "stage_id": stage.stage_id,
                                "is_fixed": bool(getattr(next_counter, "is_fixed", False)),
                            },
                        },
                        {
                            "type": "witness_reaction",
                            "stage_id": stage.stage_id,
                            "witness_id": witness_id,
                            "addressee": witness_id,
                            "statement_id": next_counter.statement_id,
                            "source": "fixed_testimony",
                            "is_fixed": bool(getattr(next_counter, "is_fixed", False)),
                            "lines": [ln.model_dump() for ln in next_lines],
                            "sfx": "sfx_gavel_1",
                        },
                        {"type": "court_record_updated"},
                    ]
                )

        return events

    async def _emit_epitaph_doctor_opinion_submission(
        self,
        session_id: str,
        stage_id: str,
        stage: TrialStage,
    ) -> list[dict[str, Any]]:
        inventory = await self.state.get_inventory(session_id)
        if "ev_ep_doctor_opinion" not in inventory:
            await self.state.add_evidence_to_inventory(session_id, "ev_ep_doctor_opinion")
        fixed_rebuttal_line = (
            ((stage.prosecution_context or {}).get("fixed_rebuttal_line"))
            or "잠깐, 양진혁씨는 평소 심장질환을 앓고 있었습니다. 의사의 소견서를 증거물로 제출합니다. 20mg은 치사량이 아니더라도 충분히 위험한 양입니다!"
        )
        return [
            {
                "type": "prosecutor_response",
                "stage_id": stage_id,
                "addressee": "pros_001",
                "response_mode": "fixed_submit",
                "mode": "fixed_submit",
                "expression_state": "basic",
                "animation_tag": "basic",
                "is_fixed": True,
                "lines": [
                    {
                        "speaker": "pros_001",
                        "dialogue": fixed_rebuttal_line,
                        "animation_tag": "basic",
                        "is_fixed": True,
                    }
                ],
            },
            {
                "type": "evidence_submitted",
                "stage_id": stage_id,
                "evidence_id": "ev_ep_doctor_opinion",
                "message": "검사가 양진혁 의사 소견서를 제출했습니다.",
                "is_fixed": True,
            },
        ]

    async def _emit_epitaph_trial2_prosecutor_opening(
        self,
        session_id: str,
        stage: TrialStage,
    ) -> list[dict[str, Any]]:
        ctx = stage.prosecution_context or {}
        submit_line = str(ctx.get("fixed_prosecutor_submit_line") or "").strip()
        if not submit_line:
            submit_line = (
                "검찰은 회사 서버 로그 기록을 증거로 제출합니다. "
                "이어서 앤서니의 노트북 기록도 제출합니다."
            )

        for evidence_id in ("ev_ep_server_log", "ev_ep_laptop"):
            inventory = await self.state.get_inventory(session_id)
            if evidence_id not in inventory:
                await self.state.add_evidence_to_inventory(session_id, evidence_id)

        prosecutor_line = ActorLine(
            speaker="pros_001",
            dialogue=submit_line,
            animation_tag="basic",
        )
        await self._persist_actor_lines(session_id, [prosecutor_line])
        events: list[dict[str, Any]] = [
            {
                "type": "prosecutor_response",
                "stage_id": stage.stage_id,
                "addressee": "pros_001",
                "response_mode": "fixed_submit",
                "mode": "fixed_submit",
                "is_fixed": True,
                "lines": [prosecutor_line.model_dump()],
            }
        ]
        for evidence_id, message in (
            ("ev_ep_server_log", "검사가 회사 서버 로그를 제출했습니다."),
            ("ev_ep_laptop", "검사가 앤서니 노트북 기록을 제출했습니다."),
        ):
            events.append(
                {
                    "type": "evidence_submitted",
                    "stage_id": stage.stage_id,
                    "evidence_id": evidence_id,
                    "message": message,
                    "is_fixed": True,
                }
            )
        return events

    async def _emit_epitaph_trial2_post_denial_script(
        self,
        session_id: str,
        stage: TrialStage,
    ) -> list[dict[str, Any]]:
        ctx = stage.prosecution_context or {}
        judge_text = str(ctx.get("fixed_judge_post_denial_line") or "").strip()
        prosecutor_text = str(ctx.get("fixed_prosecutor_explain_line") or "").strip()
        if not judge_text and not prosecutor_text:
            return []

        actor_lines: list[ActorLine] = []
        if judge_text:
            actor_lines.append(
                ActorLine(
                    speaker="judge_001",
                    dialogue=judge_text,
                    animation_tag="think",
                )
            )
        if prosecutor_text:
            actor_lines.append(
                ActorLine(
                    speaker="pros_001",
                    dialogue=prosecutor_text,
                    animation_tag="basic",
                )
            )
        await self._persist_actor_lines(session_id, actor_lines)
        return [
            {
                "type": "actor_lines",
                "stage_id": stage.stage_id,
                "is_fixed": True,
                "lines": [line.model_dump() for line in actor_lines],
            }
        ]

    async def _emit_epitaph_trial2_battle2_intro(
        self,
        session_id: str,
        stage: TrialStage,
    ) -> list[dict[str, Any]]:
        ctx = stage.prosecution_context or {}
        prosecutor_text = str(ctx.get("fixed_prosecutor_battle2_line") or "").strip()
        anthony_text = str(ctx.get("fixed_anthony_battle2_line") or "").strip()
        events: list[dict[str, Any]] = []

        inventory = await self.state.get_inventory(session_id)
        if "ev_ep_minsoo_opinion" not in inventory:
            await self.state.add_evidence_to_inventory(session_id, "ev_ep_minsoo_opinion")

        actor_lines: list[ActorLine] = []
        if prosecutor_text:
            prosecutor_line = ActorLine(
                speaker="pros_001",
                dialogue=prosecutor_text,
                animation_tag="think",
            )
            actor_lines.append(prosecutor_line)
            events.append(
                {
                    "type": "prosecutor_response",
                    "stage_id": stage.stage_id,
                    "addressee": "pros_001",
                    "response_mode": "fixed_submit",
                    "mode": "fixed_submit",
                    "is_fixed": True,
                    "lines": [prosecutor_line.model_dump()],
                }
            )
            events.append(
                {
                    "type": "evidence_submitted",
                    "stage_id": stage.stage_id,
                    "evidence_id": "ev_ep_minsoo_opinion",
                    "message": "검사가 임민수 LiDAR 소견서를 제출했습니다.",
                    "is_fixed": True,
                }
            )

        if anthony_text:
            anthony_line = ActorLine(
                speaker="def_ep_002",
                dialogue=anthony_text,
                animation_tag="serious",
            )
            actor_lines.append(anthony_line)
            events.append(
                {
                    "type": "defendant_reaction",
                    "stage_id": stage.stage_id,
                    "defendant_id": "def_ep_002",
                    "is_fixed": True,
                    "lines": [anthony_line.model_dump()],
                }
            )

        if actor_lines:
            await self._persist_actor_lines(session_id, actor_lines)
        return events

    async def _emit_epitaph_trial2_battle3_intro(
        self,
        session_id: str,
        stage: TrialStage,
    ) -> list[dict[str, Any]]:
        ctx = stage.prosecution_context or {}
        prosecutor_text = str(ctx.get("fixed_prosecutor_only_anthony_line") or "").strip()
        anthony_text = str(ctx.get("fixed_anthony_forge_line") or "").strip()
        judge_text = str(ctx.get("fixed_judge_battle3_line") or "").strip()

        inventory = await self.state.get_inventory(session_id)
        if "ev_ep_kakao" not in inventory:
            await self.state.add_evidence_to_inventory(session_id, "ev_ep_kakao")

        actor_lines: list[ActorLine] = []
        events: list[dict[str, Any]] = []

        if prosecutor_text:
            prosecutor_line = ActorLine(
                speaker="pros_001",
                dialogue=prosecutor_text,
                animation_tag="basic",
            )
            actor_lines.append(prosecutor_line)
            events.append(
                {
                    "type": "prosecutor_response",
                    "stage_id": stage.stage_id,
                    "addressee": "pros_001",
                    "response_mode": "fixed_submit",
                    "mode": "fixed_submit",
                    "is_fixed": True,
                    "lines": [prosecutor_line.model_dump()],
                }
            )

        if anthony_text:
            anthony_line = ActorLine(
                speaker="def_ep_002",
                dialogue=anthony_text,
                animation_tag="serious",
            )
            actor_lines.append(anthony_line)
            events.append(
                {
                    "type": "defendant_reaction",
                    "stage_id": stage.stage_id,
                    "defendant_id": "def_ep_002",
                    "is_fixed": True,
                    "lines": [anthony_line.model_dump()],
                }
            )

        if judge_text:
            judge_line = ActorLine(
                speaker="judge_001",
                dialogue=judge_text,
                animation_tag="think",
            )
            actor_lines.append(judge_line)
            events.append(
                {
                    "type": "judge_comment",
                    "stage_id": stage.stage_id,
                    "event_type": "battle3_open",
                    "is_fixed": True,
                    "lines": [judge_line.model_dump()],
                }
            )

        events.append(
            {
                "type": "evidence_submitted",
                "stage_id": stage.stage_id,
                "evidence_id": "ev_ep_kakao",
                "message": "검사가 앤서니·임민수 카카오 대화를 제출했습니다.",
                "is_fixed": True,
            }
        )

        if actor_lines:
            await self._persist_actor_lines(session_id, actor_lines)
        return events

    async def _emit_epitaph_trial2_stage_clear_script(
        self,
        session_id: str,
        stage: TrialStage,
    ) -> list[dict[str, Any]]:
        ctx = stage.prosecution_context or {}
        prosecutor_text = str(ctx.get("fixed_prosecutor_adjourn_request_line") or "").strip()
        judge_text = str(ctx.get("fixed_judge_adjourn_line") or "").strip()

        actor_lines: list[ActorLine] = []
        events: list[dict[str, Any]] = []

        if prosecutor_text:
            prosecutor_line = ActorLine(
                speaker="pros_001",
                dialogue=prosecutor_text,
                animation_tag="basic",
            )
            actor_lines.append(prosecutor_line)
            events.append(
                {
                    "type": "prosecutor_pressure",
                    "stage_id": stage.stage_id,
                    "intervention_type": "trial_adjourn_request",
                    "is_fixed": True,
                    "lines": [prosecutor_line.model_dump()],
                }
            )

        if judge_text:
            judge_line = ActorLine(
                speaker="judge_001",
                dialogue=judge_text,
                animation_tag="think",
            )
            actor_lines.append(judge_line)
            events.append(
                {
                    "type": "judge_comment",
                    "stage_id": stage.stage_id,
                    "event_type": "trial_adjourned",
                    "is_fixed": True,
                    "lines": [judge_line.model_dump()],
                    "sfx": "sfx_gavel_3",
                }
            )

        if actor_lines:
            await self._persist_actor_lines(session_id, actor_lines)
        return events

    def _next_counter_id(self, current_statement: Any) -> str | None:
        return getattr(current_statement, "counter_statement_id", None) or getattr(
            current_statement, "next_counter_statement_id", None
        )

    async def _add_statement_record(
        self,
        session_id: str,
        stage: TrialStage,
        statement_id: str,
        text: str,
        source: str,
        usable: bool,
    ) -> None:
        await self.state.add_court_record(
            session_id,
            CourtRecord(
                statement_id=statement_id,
                speaker=stage.active_witness_id or "witness",
                text=text,
                truth_status=TruthStatus.UNVERIFIED,
                source=source,
                usable_as_evidence=usable,
                stage_id=stage.stage_id,
            ),
        )
        if usable:
            await self.state.mark_statement_usable_as_evidence(session_id, statement_id)

    async def _clear_stage(self, session_id: str, stage: TrialStage) -> StageResult:
        ts = await self.state.get_trial_state(session_id)
        attempts = max(0, ts.stage_attempts.get(stage.stage_id, 1) - 1)
        life_lost = ts.life_lost_by_stage.get(stage.stage_id, 0)
        hints = ts.stage_hint_levels.get(stage.stage_id, 0)
        base = 100 - attempts * 8 - life_lost * 10 - hints * 5
        if ts.difficulty == "hard":
            base += 10
        max_possible = max(1, int((100 + (10 if ts.difficulty == "hard" else 0)) * stage.score_weight))
        score = min(max_possible, max(0, int(base * stage.score_weight)))
        await self.state.clear_stage(session_id, stage.stage_id, score)
        return StageResult(
            stage_id=stage.stage_id,
            cleared=True,
            failed=False,
            remaining_life=ts.stage_life,
            stage_score=score,
            max_possible_score=max_possible,
            score_ratio=round(score / max_possible, 4) if max_possible else 0.0,
            feedback="스테이지 클리어!",
        )

    def _mental_band(self, mental: int) -> str:
        if mental <= 0:
            return "breakdown"
        if mental <= 30:
            return "critical"
        if mental <= 65:
            return "shaken"
        return "steady"

    def _witness_expression(self, mental: int) -> str:
        if mental <= 0:
            return "breakdown"
        if mental <= 30:
            return "embarrassed"
        if mental <= 65:
            return "sweat"
        return "basic"

    @staticmethod
    def _line_expression(lines, default: str = "idle") -> str:
        for line in lines:
            tag = getattr(line, "animation_tag", None) or (line.get("animation_tag") if isinstance(line, dict) else None)
            if tag and tag not in ("idle", "normal", "basic"):
                return tag
        return default

    @staticmethod
    def _prosecutor_expression(response_mode: str, lines) -> str:
        explicit = FreeDialogueEngine._line_expression(lines, "")
        if explicit:
            return explicit
        return "basic"

    @staticmethod
    def _judge_expression(event_type: str, lines) -> str:
        explicit = FreeDialogueEngine._line_expression(lines, "")
        if explicit:
            return explicit
        if event_type in ("objection_sustained", "stage_cleared"):
            return "success"
        if event_type in ("stage_failed", "life_lost"):
            return "serious"
        return "think"

    def _should_verify_actor_output(self, stage: TrialStage) -> bool:
        return stage.stage_id == "stage_epitaph_car" and stage.stage_type == StageType.VS_WITNESS

    async def _verify_actor_response(
        self,
        *,
        role: str,
        stage: TrialStage,
        stage_phase: str,
        user_text: str,
        response,
        current_statement: dict[str, Any] | None,
        turn_batch_lines: list[dict[str, Any]],
        forbidden_claims: list[str],
        inventory_evidence: list[dict[str, Any]],
        retry_fn: Callable[[str], Awaitable[Any]],
        fallback_speaker: str,
    ) -> tuple[Any, dict[str, Any]]:
        if not self._should_verify_actor_output(stage):
            return response, {"applied": False, "valid": True, "issues": []}

        try:
            result = await self.response_verifier.verify(
                role=role,
                stage_id=stage.stage_id,
                stage_phase=stage_phase,
                response=response,
                user_text=user_text,
                current_statement=current_statement,
                turn_batch_lines=turn_batch_lines,
                forbidden_claims=forbidden_claims,
                inventory_evidence=inventory_evidence,
            )
        except Exception as exc:
            self.logger.warning(
                "Verifier failed for initial %s response (stage=%s): %s",
                role,
                stage.stage_id,
                exc,
            )
            return response, {"applied": True, "valid": True, "issues": ["verifier_error"], "skipped": True}
        if result.valid:
            return response, {"applied": True, "valid": True, "issues": []}

        retry_feedback = result.suggested_fix or "역할을 유지하고 자기모순 없이 다시 답하세요."
        try:
            retried = await retry_fn(retry_feedback)
        except Exception:
            retried = response
        try:
            retry_result = await self.response_verifier.verify(
                role=role,
                stage_id=stage.stage_id,
                stage_phase=stage_phase,
                response=retried,
                user_text=user_text,
                current_statement=current_statement,
                turn_batch_lines=turn_batch_lines,
                forbidden_claims=forbidden_claims,
                inventory_evidence=inventory_evidence,
            )
        except Exception as exc:
            self.logger.warning(
                "Verifier failed for retried %s response (stage=%s): %s",
                role,
                stage.stage_id,
                exc,
            )
            return retried, {
                "applied": True,
                "valid": True,
                "retried": True,
                "issues": result.issues + ["verifier_error_after_retry"],
                "skipped": True,
            }
        if retry_result.valid:
            return retried, {"applied": True, "valid": True, "issues": result.issues, "retried": True}

        fallback = self._verification_fallback_line(
            role=role,
            speaker=fallback_speaker,
            current_statement=current_statement,
        )
        return (
            fallback,
            {
                "applied": True,
                "valid": False,
                "retried": True,
                "issues": result.issues + retry_result.issues,
                "used_fallback": True,
            },
        )

    def _verification_fallback_line(
        self,
        *,
        role: str,
        speaker: str,
        current_statement: dict[str, Any] | None,
    ) -> ActorResponse:
        statement_text = (current_statement or {}).get("text", "")
        if role == "prosecutor":
            dialogue = (
                f"검찰은 증인의 핵심 진술을 유지합니다. {statement_text[:72]}"
                if statement_text
                else "검찰은 증인의 핵심 진술을 유지하며 변호인의 공격을 받아들일 수 없습니다."
            )
            return ActorResponse(lines=[ActorLine(speaker=speaker, dialogue=dialogue, animation_tag="basic")])
        if role == "witness":
            dialogue = (
                statement_text[:120]
                if statement_text
                else "제가 본 사실은 방금까지의 진술과 같습니다."
            )
            return ActorResponse(lines=[ActorLine(speaker=speaker, dialogue=dialogue, animation_tag="basic")])
        return ActorResponse(
            lines=[
                ActorLine(
                    speaker=speaker,
                    dialogue="현재 제출된 주장과 증거의 연결만 판단하겠습니다.",
                    animation_tag="think",
                )
            ]
        )

    async def _build_judge_response(
        self,
        session_id: str,
        episode: EpisodeData,
        stage: TrialStage,
        text: str,
        selected_evidence_ids: list[str],
        judge_trigger: str,
        mode: str,
    ) -> list[dict[str, Any]]:
        ts = await self.state.get_trial_state(session_id)
        current_statement = self._current_statement(stage, ts)
        current_statement_dict = (
            current_statement.model_dump() if current_statement and hasattr(current_statement, "model_dump") else None
        )
        selected_details = await self._selected_evidence_details(session_id, episode, selected_evidence_ids)
        court_records = [record.model_dump() for record in await self.state.get_court_records(session_id)]

        if stage.stage_type == StageType.VS_WITNESS and current_statement:
            evaluation = await self.evaluator.evaluate_stage_argument(
                stage_type=stage.stage_type.value,
                current_stage=stage,
                current_statement=current_statement,
                user_text=text,
                selected_evidence_ids=selected_evidence_ids,
                selected_evidence_details=selected_details,
                court_records=court_records,
            )
        else:
            evaluation = DefenseArgumentEvaluation(
                relevance=RelevanceLevel.PARTIALLY_RELEVANT if text.strip() else RelevanceLevel.IRRELEVANT,
                core_match_score=0.4 if text.strip() else 0.1,
                logic_score=0.4 if selected_evidence_ids else 0.2,
                evidence_usage_score=0.5 if selected_evidence_ids else 0.0,
                verdict=AnswerVerdict.PARTIAL_SUCCESS if selected_evidence_ids else AnswerVerdict.FAIL,
                reason=self._fallback_evaluation_reason(text, selected_evidence_ids, judge_trigger),
            )

        if (
            current_statement_dict
            and current_statement_dict.get("statement_id") == "stmt_epitaph_isoeun_4"
            and AnswerEvaluatorLLM.epitaph_battle4_is_clear_success(text, selected_evidence_ids)
            and evaluation.verdict != AnswerVerdict.SUCCESS
        ):
            evaluation = evaluation.model_copy(
                update={
                    "verdict": AnswerVerdict.SUCCESS,
                    "relevance": RelevanceLevel.RELEVANT,
                    "reason": "좋습니다. VX 정보와 증언 #4를 연결해 피부 치사량·춤 행동 모순을 지적했습니다.",
                }
            )

        witness_mental = ts.witness_mental_by_stage.get(stage.stage_id, stage.witness_mental)
        judge_persuasion = ts.judge_persuasion_by_stage.get(stage.stage_id, stage.judge_persuasion)

        if mode == "objection" or judge_trigger == "objection":
            judge_event_type = self._objection_event_type(evaluation.verdict)
        elif judge_trigger == "decisive":
            judge_event_type = self._decisive_event_type(evaluation.verdict)
        elif judge_trigger == "passive":
            judge_event_type = "passive_intervention"
        elif judge_trigger == "direct_address":
            judge_event_type = "passive_intervention"
        else:
            judge_event_type = self._decisive_event_type(evaluation.verdict)

        judge = await self.judge_actor.evaluate_free_dialogue(
            stage_type=stage.stage_type.value,
            stage_id=stage.stage_id,
            event_type=judge_event_type,
            evaluation=evaluation,
            user_text=text,
            current_statement=current_statement_dict,
            selected_evidence_ids=selected_evidence_ids,
            trigger=judge_trigger,
            remaining_life=ts.stage_life,
            witness_mental=witness_mental,
            judge_persuasion=judge_persuasion,
        )
        judge_verification = {"applied": False, "valid": True, "issues": []}
        if self._should_verify_actor_output(stage):
            judge, judge_verification = await self._verify_actor_response(
                role="judge",
                stage=stage,
                stage_phase=ts.stage_phase,
                user_text=text,
                response=judge,
                current_statement=current_statement_dict,
                turn_batch_lines=[],
                forbidden_claims=episode.forbidden_claims,
                inventory_evidence=selected_details,
                retry_fn=lambda feedback: self.judge_actor.evaluate_free_dialogue(
                    stage_type=stage.stage_type.value,
                    stage_id=stage.stage_id,
                    event_type=judge_event_type,
                    evaluation=evaluation,
                    user_text=text,
                    current_statement=current_statement_dict,
                    selected_evidence_ids=selected_evidence_ids,
                    trigger=judge_trigger,
                    remaining_life=ts.stage_life,
                    witness_mental=witness_mental,
                    judge_persuasion=judge_persuasion,
                    verifier_feedback=feedback,
                ),
                fallback_speaker="judge_001",
            )

        await self._persist_actor_lines(session_id, judge.lines)

        events: list[dict[str, Any]] = [
            {
                "type": "defense_argument_evaluated",
                "stage_id": stage.stage_id,
                "evaluation": evaluation.model_dump(),
                "mode": mode if mode == "objection" else "free_dialogue",
                "selected_evidence_ids": selected_evidence_ids,
                "judge_comment": evaluation.reason,
                "trigger": judge_trigger,
            },
            {
                "type": "judge_intervention",
                "event_type": judge_event_type,
                "trigger": judge_trigger,
                "expression_state": self._judge_expression(judge_event_type, judge.lines),
                "animation_tag": self._judge_expression(judge_event_type, judge.lines),
                "lines": [ln.model_dump() for ln in judge.lines],
                "judge_comment": evaluation.reason,
                "sfx": "sfx_gavel_1",
                "verification": judge_verification,
            },
        ]

        if self._judge_ruling_costs_life(judge_event_type):
            life_loss = self._default_life_loss(stage, ts)
            if life_loss > 0 and ts.stage_life > 0:
                events.extend(
                    await self._append_life_penalty_events(session_id, stage.stage_id, life_loss)
                )

        if (
            mode == "objection"
            and stage.stage_type == StageType.VS_WITNESS
            and evaluation.verdict not in (AnswerVerdict.SUCCESS, AnswerVerdict.PARTIAL_SUCCESS)
            and not (
                (current_statement_dict or {}).get("statement_id") == "stmt_epitaph_isoeun_4"
                and AnswerEvaluatorLLM.epitaph_battle4_is_clear_success(text, selected_evidence_ids)
            )
        ):
            events.extend(
                await self._emit_objection_failure_witness(
                    session_id,
                    stage,
                    evaluation,
                    text,
                    selected_evidence_ids,
                    current_statement_dict,
                )
            )

        return events

    async def _emit_objection_failure_witness(
        self,
        session_id: str,
        stage: TrialStage,
        evaluation: DefenseArgumentEvaluation,
        user_text: str,
        selected_evidence_ids: list[str],
        current_statement: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        witness_id = stage.active_witness_id or "wit_ep_001"
        ts = await self.state.get_trial_state(session_id)
        witness_mental = ts.witness_mental_by_stage.get(stage.stage_id, stage.witness_mental)
        witness = await self.witness_actor.generate_stage_reaction(
            event_type="argument_fail",
            witness_id=witness_id,
            evaluation=evaluation,
            current_statement=current_statement,
            user_answer=user_text,
            selected_evidence_ids=selected_evidence_ids,
            witness_mental=witness_mental,
        )
        await self._persist_actor_lines(session_id, witness.lines)
        return [
            {
                "type": "witness_reaction",
                "stage_id": stage.stage_id,
                "witness_id": witness_id,
                "addressee": witness_id,
                "source": "objection_fail",
                "expression_state": self._line_expression(witness.lines, "basic"),
                "animation_tag": self._line_expression(witness.lines, "basic"),
                "lines": [ln.model_dump() for ln in witness.lines],
                "is_fixed": False,
            }
        ]

    @staticmethod
    def _judge_ruling_costs_life(judge_event_type: str) -> bool:
        return judge_event_type in {"argument_fail", "objection_overruled", "objection_rejected"}

    @staticmethod
    def _default_life_loss(stage: TrialStage, ts: Any) -> int:
        current_statement = None
        if ts.current_counter_statement_id:
            current_statement = stage.counter_by_id(ts.current_counter_statement_id)
        elif ts.current_testimony_id:
            current_statement = stage.testimony_by_id(ts.current_testimony_id)
        elif stage.fixed_testimony_chain:
            current_statement = stage.fixed_testimony_chain[0]
        if current_statement is not None:
            return max(1, getattr(current_statement, "life_loss_on_fail", 1))
        return 1

    async def _append_life_penalty_events(
        self,
        session_id: str,
        stage_id: str,
        life_loss: int,
    ) -> list[dict[str, Any]]:
        remaining = await self.state.apply_life_loss(session_id, life_loss)
        events: list[dict[str, Any]] = [
            {
                "type": "life_update",
                "stage_id": stage_id,
                "remaining_life": remaining,
                "life_loss": life_loss,
            }
        ]
        if remaining <= 0:
            await self.state.fail_stage(session_id, stage_id)
            events.append(
                {
                    "type": "stage_failed",
                    "stage_id": stage_id,
                    "remaining_life": 0,
                    "feedback": "생명이 모두 소진되었습니다. 스테이지를 다시 시작하세요.",
                }
            )
        return events

    @staticmethod
    def _responder_addressee(responder: str, stage: TrialStage, primary_addressee: str) -> str:
        if responder == "prosecutor":
            return "pros_001"
        if responder in {"witness", "witness_followup"}:
            return stage.active_witness_id or primary_addressee
        return primary_addressee

    @staticmethod
    def _lines_to_batch_entries(lines, speaker: str) -> list[dict[str, str]]:
        entries = []
        for line in lines:
            if line.dialogue:
                entries.append({"speaker": speaker, "text": line.dialogue})
        return entries

    async def _persist_actor_lines(self, session_id: str, lines) -> None:
        for line in lines:
            if line.dialogue:
                await self.state.append_dialogue(session_id, line.speaker, line.dialogue)

        if not lines:
            return

        ts = await self.state.get_trial_state(session_id)
        response_text = " ".join(ln.dialogue for ln in lines if ln.dialogue).strip()
        if response_text:
            speaker = lines[0].speaker
            ts.free_dialogue_history = ts.free_dialogue_history + [
                {"speaker": speaker, "text": response_text, "addressee": "player"}
            ]
            await self.state.save_trial_state(session_id, ts)

    def _objection_event_type(self, verdict: AnswerVerdict) -> str:
        if verdict == AnswerVerdict.SUCCESS:
            return "objection_sustained"
        if verdict == AnswerVerdict.PARTIAL_SUCCESS:
            return "objection_partial"
        if verdict == AnswerVerdict.IRRELEVANT:
            return "objection_overruled"
        return "objection_overruled"

    def _decisive_event_type(self, verdict: AnswerVerdict) -> str:
        if verdict == AnswerVerdict.SUCCESS:
            return "argument_success"
        if verdict == AnswerVerdict.PARTIAL_SUCCESS:
            return "partial_success"
        return "argument_fail"

    def _fallback_evaluation_reason(
        self, text: str, selected_evidence_ids: list[str], judge_trigger: str
    ) -> str:
        if judge_trigger == "objection":
            if selected_evidence_ids:
                return "이의에 대한 판사 평가: 제시 증거와 주장의 연결을 검토했습니다."
            return "이의에 대한 판사 평가: 증거 없이는 증언의 모순을 입증하기 어렵습니다."
        if not text.strip():
            return "내용이 비어 있어 평가할 수 없습니다."
        return "현재 쟁점과의 관련성을 기준으로 판사 평가를 수행했습니다."

    async def _build_free_dialogue_context(
        self,
        session_id: str,
        episode: EpisodeData,
        stage: TrialStage,
        addressee: str,
        ts: Any,
    ) -> dict[str, Any]:
        snapshot = await self.state.get_snapshot(session_id)
        character_knowledge = self.knowledge.build_context(addressee, episode, snapshot)
        dialogue_history = ts.free_dialogue_history[-12:] or await self.state.get_dialogue_history(
            session_id, limit=12
        )

        current_statement = self._current_statement(stage, ts)
        current_statement_dict = (
            current_statement.model_dump() if current_statement and hasattr(current_statement, "model_dump") else None
        )

        inventory_details = []
        for evidence_id in snapshot.inventory:
            evidence = episode.get_evidence(evidence_id)
            if evidence:
                inventory_details.append(evidence.model_dump())

        truth = episode.absolute_truth or {}
        return {
            "role": self._character_def(episode, addressee).get("role", "unknown"),
            "character": self._character_def(episode, addressee),
            "character_knowledge": character_knowledge,
            "dialogue_history": dialogue_history,
            "exchange_count": ts.free_dialogue_exchanges,
            "episode_title": episode.title,
            "case_summary": {
                "defendant": truth.get("defendant"),
                "victim": truth.get("victim"),
                "date": truth.get("date"),
                "location": truth.get("location"),
                "time_of_death": truth.get("time_of_death"),
            },
            "prosecution_case": episode.prosecution_case.model_dump() if episode.prosecution_case else {},
            "forbidden_claims": episode.forbidden_claims,
            "allowed_lies": episode.allowed_lies if addressee != "pros_001" else [],
            "stage_context": {
                "stage_id": stage.stage_id,
                "stage_type": stage.stage_type.value,
                "prosecution_context": stage.prosecution_context,
                "active_witness_id": stage.active_witness_id,
                "current_statement": current_statement_dict,
                "fixed_testimony_chain": [t.model_dump() for t in stage.fixed_testimony_chain],
            },
            "inventory_evidence": inventory_details,
            "witness_mental": ts.witness_mental_by_stage.get(stage.stage_id, stage.witness_mental),
        }

    def _character_def(self, episode: EpisodeData, char_id: str) -> dict[str, Any]:
        for character in episode.characters.values():
            if character.id == char_id:
                return character.model_dump()
        return {"id": char_id, "name": char_id, "role": "unknown"}

    def _current_statement(self, stage: TrialStage, ts):
        if ts.current_counter_statement_id:
            return stage.counter_by_id(ts.current_counter_statement_id)
        if ts.current_testimony_id:
            return stage.testimony_by_id(ts.current_testimony_id)
        if stage.fixed_testimony_chain:
            return stage.fixed_testimony_chain[0]
        return None

    async def _selected_evidence_details(
        self, session_id: str, episode: EpisodeData, selected_evidence_ids: list[str]
    ) -> list[dict[str, Any]]:
        records = {record.statement_id: record for record in await self.state.get_court_records(session_id)}
        details = []
        for evidence_id in selected_evidence_ids[:2]:
            evidence = episode.get_evidence(evidence_id)
            if evidence:
                details.append(evidence.model_dump())
            elif evidence_id in records:
                record = records[evidence_id]
                details.append(
                    {
                        "id": record.statement_id,
                        "name": f"발언 기록: {record.speaker}",
                        "description": record.text,
                        "source": record.source,
                    }
                )
        return details
