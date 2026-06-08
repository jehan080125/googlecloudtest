import pytest

from backend.ai_services.dialogue_router import (
    CROSS_EXAM_TURN_DEPTH,
    detect_decisive_point,
    orchestrate_player_turn,
    route_addressee,
    should_trigger_passive_judge,
)
from backend.ai_services.prosecutor_actor import ProsecutorActorLLM
from backend.core.response_verifier import ResponseVerifierLLM
from backend.core.court_orchestrator import CourtOrchestrator
from backend.core.state_manager import StateManager
from backend.schemas.court import ActorLine, ActorResponse
from backend.services.episode_loader import load_episode

STAGE_ID = "stage_yamano_chain"
EPITAPH_FIXED_OPENING_JUDGE = "피고, 범행 사실을 인정하시나요?"
EPITAPH_FIXED_OPENING_DEFENDANT = "아니요, 인정하지 않습니다!"
EPITAPH_FIXED_STMT_1 = (
    "클럽에서 처음 만난 양진혁가 저에게 술을 사줬어요. 바텐더가 준 술을 함께 마시고, 춤을 췄는데, "
    "갑자기 제가 정신을 잃고 쓰러졌습니다! 일어나 보니 병원이었고, 양진혁가 죽어있었어요.... "
    "바텐더가 준 술에 독이 들어있었던 겁니다!"
)
EPITAPH_FIXED_STMT_2 = "앗 그건 양진혁이 저에게 쓰러지면서 제 손에 독이 든 술이 묻어서 그렇습니다"
EPITAPH_FIXED_STMT_3 = "제가 충격이 심한 상황이라, 기억을 잃었던 것 같습니다."
EPITAPH_FIXED_STMT_4 = (
    "실은 술을 조금만 마셨는데, 술에 잔뜩 취한 양진혁 씨가 비틀거리다가 제 술잔에 있던 술을 제 오른손에 모두 쏟았어요! "
    "그래서 손에 술이 묻은 상태로 춤을 추다가... 그 뒤에 독 기운이 올라와서 제가 먼저 쓰러진 겁니다!"
)
EPITAPH_FIXED_PROSECUTOR_REBUTTAL = (
    "잠깐, 양진혁씨는 평소 심장질환을 앓고 있었습니다. 의사의 소견서를 증거물로 제출합니다. "
    "20mg은 치사량이 아니더라도 충분히 위험한 양입니다!"
)
EPITAPH_FIXED_JUDGE_CROSS = "변호인 할 말 있습니까?"
EPITAPH_FIXED_DEFENSE_CROSS = "증인에게 물어보고 싶은 것이 있습니다."
EPITAPH_FIXED_CONFESSION = (
    "(머리를 감싸 쥐며 부르르 떤다) 으, 으으으... 기, 기건...! 내가... 내가 피부가 엄청나게 두꺼워서...! "
    "아니, 장갑을 끼고 있어서...!! 윽 ...! 사실 다 거짓말이야. 나를 용서해줘. 바텐더는 죄가 없어…. (죄를 시인한다.)"
)
EPITAPH_FIXED_PROSECUTOR_ADJOURN = "앗! 좀 더 수사가 필요할 것 같습니다! 재판을 멈춰주세요!"
EPITAPH_FIXED_JUDGE_ADJOURN = "알겠습니다. 수사가 완료될 때 까지 재판을 연기하도록 하죠."
EPITAPH_MODEL_ANSWER_1 = (
    "VX 정보 50mg·술잔 20mg, 마시기만으론 치사량 미달. 독살 단정은 성급."
)
EPITAPH_MODEL_ANSWER_2 = "오른손에서 VX물질이 발견된 이유에 대해서 설명해 주세요!"
EPITAPH_MODEL_ANSWER_3 = (
    "증언 #1·#2: 먼저 기절 vs 양진혁이 쓰러질 때 손에 묻음. 순서 모순!"
)
EPITAPH_MODEL_ANSWER_4 = (
    "춤 전 손에 치사량 독이 쏟아졌다면 즉사! VX정보 피부 치사량 10mg입니다!"
)
EPITAPH_BATTLE4_USER_TEXT = EPITAPH_MODEL_ANSWER_4

EPITAPH_TRIAL2_OPENING_JUDGE = "피고 앤서니는 범행 사실을 인정합니까?"
EPITAPH_TRIAL2_STMT = (
    "앤서니가 회사 서버를 통해 연행 호송차에 급가속 후 우회전 명령을 보냈습니다. "
    "노트북·서버 로그가 그를 가리킵니다."
)
EPITAPH_TRIAL2_MODEL_1 = "CCTV 우회전·서버로그 좌회전—방향 모순!"
EPITAPH_TRIAL2_MODEL_2 = "소견 희박·살의면 우회전 코드 보냈을 것. 로그는 좌회전."
EPITAPH_TRIAL2_MODEL_3 = "카카오는 소호 보복 의심뿐. 이소은 살해 동기 증명 안 됨—아직 무죄."
EPITAPH_TRIAL2_BATTLE1_USER = EPITAPH_TRIAL2_MODEL_1
EPITAPH_TRIAL2_BATTLE2_USER = EPITAPH_TRIAL2_MODEL_2
EPITAPH_TRIAL2_BATTLE3_USER = EPITAPH_TRIAL2_MODEL_3


@pytest.fixture
def episode():
    return load_episode("turnabout_clock")


def test_route_addressee_to_prosecutor(episode):
    stage = episode.get_stage(STAGE_ID)
    assert route_addressee("검사님, 그 논리는 맞습니까?", stage) == "pros_001"


def test_route_addressee_to_witness(episode):
    stage = episode.get_stage(STAGE_ID)
    assert route_addressee("증인, 정말 2시에 봤습니까?", stage) == "wit_001"


def test_route_addressee_defaults_to_witness_on_vs_witness(episode):
    stage = episode.get_stage(STAGE_ID)
    assert route_addressee("그때 무엇을 보셨나요?", stage) == "wit_001"


def test_detect_decisive_point_keyword():
    assert detect_decisive_point("부검 기록과 증언이 모순됩니다") is True
    assert detect_decisive_point("그때 날씨는 어땠나요?") is False


def test_orchestrate_objection_always_judge(episode):
    stage = episode.get_stage(STAGE_ID)
    plan = orchestrate_player_turn("이의!", stage, 1, "objection")
    assert plan.responders == ["judge"]
    assert plan.trigger_judge_evaluation is True
    assert plan.judge_trigger == "objection"


def test_orchestrate_cross_exam_free_includes_prosecutor(episode):
    stage = episode.get_stage(STAGE_ID)
    plan = orchestrate_player_turn(
        "증인, 그때 정말 2시였습니까?",
        stage,
        1,
        "question",
        stage_phase="cross_exam_free",
    )
    assert plan.responders == ["witness", "prosecutor", "witness_followup"]
    assert plan.judge_trigger == "none"


def test_orchestrate_cross_exam_turn_depth_configurable(episode):
    stage = episode.get_stage(STAGE_ID)
    shallow = orchestrate_player_turn(
        "증인, 그때 정말 2시였습니까?",
        stage,
        1,
        "question",
        stage_phase="cross_exam_free",
        turn_depth=2,
    )
    assert shallow.responders == ["witness", "prosecutor"]
    assert CROSS_EXAM_TURN_DEPTH == 3


def test_orchestrate_cross_exam_free_prosecutor_only_when_addressed(episode):
    stage = episode.get_stage(STAGE_ID)
    plan = orchestrate_player_turn(
        "검사님, 그 논리는 맞습니까?",
        stage,
        1,
        "question",
        stage_phase="cross_exam_free",
    )
    assert plan.responders == ["prosecutor"]


def test_orchestrate_decisive_includes_judge(episode):
    stage = episode.get_stage(STAGE_ID)
    plan = orchestrate_player_turn("부검과 증언이 모순됩니다!", stage, 1, "question")
    assert plan.responders == ["witness", "judge"]
    assert plan.judge_trigger == "decisive"


def test_orchestrate_decisive_cross_exam_chains_before_judge(episode):
    stage = episode.get_stage(STAGE_ID)
    plan = orchestrate_player_turn(
        "부검과 증언이 모순됩니다!",
        stage,
        1,
        "question",
        stage_phase="cross_exam_free",
    )
    assert plan.responders == ["witness", "prosecutor", "witness_followup", "judge"]
    assert plan.judge_trigger == "decisive"


def test_orchestrate_passive_cross_exam_appends_judge(episode):
    stage = episode.get_stage(STAGE_ID)
    plan = orchestrate_player_turn(
        "그때 어디 계셨나요?",
        stage,
        3,
        "question",
        stage_phase="cross_exam_free",
    )
    assert plan.responders[-1] == "judge"
    assert plan.judge_trigger == "passive"


def test_passive_judge_trigger_by_keyword():
    assert should_trigger_passive_judge("판사님, 허락해 주십시오.", 1) is True


def test_passive_judge_trigger_every_third_exchange():
    assert should_trigger_passive_judge("그때 어디 계셨나요?", 3) is True
    assert should_trigger_passive_judge("그때 어디 계셨나요?", 2) is False


@pytest.mark.asyncio
async def test_free_dialogue_question_routes_to_witness():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "증인, 그때 정말 2시였습니까?",
        "question",
    )
    types = [e["type"] for e in events]
    assert "orchestration_planned" in types
    assert "addressee_routed" in types
    assert "witness_reaction" in types
    assert "orchestration_complete" in types
    routed = next(e for e in events if e["type"] == "addressee_routed")
    assert routed["addressee"] == "wit_001"


@pytest.mark.asyncio
async def test_free_dialogue_objection_triggers_judge():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await state.add_evidence(sid, "autopsy_report")
    await court.start_court(sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순됩니다!",
        "objection",
        ["autopsy_report"],
    )
    types = [e["type"] for e in events]
    assert "defense_argument_evaluated" in types
    assert "judge_intervention" in types
    judge = next(e for e in events if e["type"] == "judge_intervention")
    assert judge["trigger"] == "objection"
    assert judge["lines"]
    assert judge["lines"][0]["speaker"] == "judge_001"
    evaluated = next(e for e in events if e["type"] == "defense_argument_evaluated")
    assert evaluated.get("judge_comment")


@pytest.mark.asyncio
async def test_decisive_contradiction_routes_judge_after_witness():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "증인, 부검 기록과 모순되지 않습니까?",
        "question",
    )
    types = [e["type"] for e in events]
    assert types.index("witness_reaction") < types.index("judge_intervention")
    judge = next(e for e in events if e["type"] == "judge_intervention")
    assert judge["trigger"] == "decisive"


@pytest.mark.asyncio
async def test_passive_judge_after_three_exchanges():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    for idx in range(3):
        events = await court.process_free_dialogue(
            sid,
            STAGE_ID,
            f"질문 {idx + 1}: 그때 무엇을 보셨나요?",
            "question",
        )
    assert any(e["type"] == "judge_intervention" for e in events)
    passive = next(e for e in events if e["type"] == "judge_intervention")
    assert passive["trigger"] == "passive"


@pytest.mark.asyncio
async def test_free_dialogue_witness_response_uses_contextual_mock():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "증인, 정말 2시에 봤습니까?",
        "question",
    )
    reaction = next(e for e in events if e["type"] == "witness_reaction")
    assert reaction["lines"]
    assert reaction["lines"][0]["speaker"] == "wit_001"
    assert "2시" in reaction["lines"][0]["dialogue"]


@pytest.mark.asyncio
async def test_free_dialogue_prosecutor_response_uses_contextual_mock(episode):
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "검사님, 검찰은 왜 유죄를 주장합니까?",
        "question",
    )
    response = next(e for e in events if e["type"] == "prosecutor_response")
    assert response["lines"]
    assert response["lines"][0]["speaker"] == "pros_001"
    assert "검찰" in response["lines"][0]["dialogue"]


@pytest.mark.asyncio
async def test_free_dialogue_appends_actor_lines_to_history():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    await court.process_free_dialogue(sid, STAGE_ID, "증인, 그때 어디 계셨나요?", "question")
    history = await state.get_dialogue_history(sid, limit=10)
    speakers = [entry["speaker"] for entry in history]
    assert "player" in speakers
    assert "wit_001" in speakers


@pytest.mark.asyncio
async def test_free_dialogue_history_persisted_in_trial_state():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    await court.process_free_dialogue(sid, STAGE_ID, "증인, 2시에 봤습니까?", "question")
    await court.process_free_dialogue(sid, STAGE_ID, "그때 방 안은 어땠습니까?", "question")

    ts = await state.get_trial_state(sid)
    assert ts.free_dialogue_exchanges == 2
    assert len(ts.free_dialogue_history) >= 4
    player_turns = [entry for entry in ts.free_dialogue_history if entry["speaker"] == "player"]
    assert len(player_turns) == 2
    assert player_turns[0]["text"] == "증인, 2시에 봤습니까?"
    assert player_turns[1]["addressee"] == "wit_001"


@pytest.mark.asyncio
async def test_free_dialogue_context_includes_knowledge_scope(episode):
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM
    from backend.ai_services.judge_actor import JudgeActorLLM
    from backend.ai_services.prosecutor_actor import ProsecutorActorLLM
    from backend.ai_services.witness_actor import WitnessActorLLM
    from backend.core.free_dialogue_engine import FreeDialogueEngine

    state = StateManager()
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    ts = await state.get_trial_state(sid)
    stage = episode.get_stage(STAGE_ID)

    engine = FreeDialogueEngine(
        state,
        AnswerEvaluatorLLM(None),
        WitnessActorLLM(None),
        ProsecutorActorLLM(None),
        JudgeActorLLM(None),
    )
    context = await engine._build_free_dialogue_context(sid, episode, stage, "wit_001", ts)

    assert context["role"] == "witness"
    assert context["character"]["id"] == "wit_001"
    assert "character_knowledge" in context
    assert context["character_knowledge"]["scope"] == ["own_statements"]
    assert "case_summary" in context
    assert context["case_summary"]["victim"] == "타카비 미카"
    assert "allowed_lies" in context
    assert len(context["allowed_lies"]) > 0
    assert "stage_context" in context
    assert context["stage_context"]["current_statement"] is not None


@pytest.mark.asyncio
async def test_objection_success_triggers_cross_exam_phase():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await state.add_evidence(sid, "autopsy_report")
    await court.start_court(sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순됩니다! 4시 사망인데 2시에 시체를 봤다고?",
        "objection",
        ["autopsy_report"],
    )
    types = [e["type"] for e in events]
    assert "defense_argument_evaluated" in types
    evaluated = next(e for e in events if e["type"] == "defense_argument_evaluated")
    assert evaluated["evaluation"]["verdict"] == "success"
    assert "phase_transition" in types
    transition = next(e for e in events if e["type"] == "phase_transition")
    assert transition["to_phase"] == "cross_exam_free"
    fixed = next(
        e for e in events if e["type"] == "witness_reaction" and e.get("source") == "fixed_testimony"
    )
    assert "TV" in fixed["lines"][0]["dialogue"] or "비디오" in fixed["lines"][0]["dialogue"]

    ts = await state.get_trial_state(sid)
    assert ts.stage_phase == "cross_exam_free"
    assert ts.current_counter_statement_id == "counter_yamano_tv_sound"


@pytest.mark.asyncio
async def test_objection_fail_stays_in_testimony_phase():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "그냥 이상해요.",
        "objection",
        [],
    )
    evaluated = next(e for e in events if e["type"] == "defense_argument_evaluated")
    assert evaluated["evaluation"]["verdict"] in ("fail", "irrelevant")
    assert not any(e["type"] == "phase_transition" for e in events)

    ts = await state.get_trial_state(sid)
    assert ts.stage_phase == "testimony"


def test_prosecutor_mock_defend_witness_does_not_undermine():
    actor = ProsecutorActorLLM(None)
    response = actor._mock_free_question(
        "증인, 정말 2시에 봤습니까?",
        {
            "response_mode": "defend_witness",
            "turn_batch_lines": [{"speaker": "wit_001", "text": "오후 2시에 신고했습니다."}],
            "stage_context": {"prosecution_context": {}},
            "prosecution_case": {"fixed_claim_pool": [{"summary": "피고인의 현장 관련성"}]},
        },
    )
    dialogue = " ".join(line.dialogue for line in response.lines)
    assert not ProsecutorActorLLM.undermines_witness(dialogue)
    assert "변호인" in dialogue or "증언" in dialogue


@pytest.mark.asyncio
async def test_cross_exam_free_generates_multiple_events_per_turn():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await state.add_evidence(sid, "autopsy_report")
    await court.start_court(sid)

    await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순됩니다! 4시 사망인데 2시에 시체를 봤다고?",
        "objection",
        ["autopsy_report"],
    )

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "증인, 정말 2시에 봤습니까?",
        "question",
    )
    reaction_types = [
        e["type"]
        for e in events
        if e["type"] in {"witness_reaction", "prosecutor_response"}
    ]
    assert len(reaction_types) >= 2
    assert reaction_types.index("witness_reaction") < reaction_types.index("prosecutor_response")
    witness_events = [e for e in events if e["type"] == "witness_reaction"]
    assert len(witness_events) >= 2
    assert any(e.get("source") == "witness_followup" for e in witness_events)


@pytest.mark.asyncio
async def test_cross_exam_free_question_routes_witness_and_prosecutor():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await state.add_evidence(sid, "autopsy_report")
    await court.start_court(sid)

    await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순됩니다! 4시 사망인데 2시에 시체를 봤다고?",
        "objection",
        ["autopsy_report"],
    )

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "증인, 정말 2시에 봤습니까?",
        "question",
    )
    types = [e["type"] for e in events]
    assert "witness_reaction" in types
    assert "prosecutor_pressure" in types or "prosecutor_response" in types
    planned = next(e for e in events if e["type"] == "orchestration_planned")
    assert planned["stage_phase"] == "cross_exam_free"
    assert planned["responders"] == ["witness", "prosecutor", "witness_followup"]
    prosecutor = next(e for e in events if e["type"] == "prosecutor_response")
    assert prosecutor.get("response_mode") == "defend_witness"
    dialogue = prosecutor["lines"][0]["dialogue"]
    assert not ProsecutorActorLLM.undermines_witness(dialogue)


@pytest.mark.asyncio
async def test_judge_success_emits_fixed_counter_from_episode():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await state.add_evidence(sid, "autopsy_report")
    await court.start_court(sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순됩니다! 4시 사망인데 2시에 시체를 봤다고?",
        "objection",
        ["autopsy_report"],
    )
    judge_idx = next(i for i, e in enumerate(events) if e["type"] == "judge_intervention")
    fixed_idx = next(
        i for i, e in enumerate(events) if e["type"] == "witness_reaction" and e.get("source") == "fixed_testimony"
    )
    assert judge_idx < fixed_idx
    fixed = events[fixed_idx]
    assert fixed["statement_id"] == "counter_yamano_tv_sound"
    assert "TV" in fixed["lines"][0]["dialogue"] or "비디오" in fixed["lines"][0]["dialogue"]


@pytest.mark.asyncio
async def test_second_objection_success_advances_counter_chain():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await state.add_evidence(sid, "autopsy_report")
    await state.add_evidence(sid, "outage_record")
    await court.start_court(sid)

    await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순됩니다! 4시 사망인데 2시에 시체를 봤다고?",
        "objection",
        ["autopsy_report"],
    )

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "정전 기록에 따르면 TV나 비디오는 작동할 수 없습니다! 정전 중 모순입니다!",
        "objection",
        ["outage_record"],
    )
    fixed = next(
        e for e in events if e["type"] == "witness_reaction" and e.get("source") == "fixed_testimony"
    )
    assert fixed["statement_id"] == "counter_yamano_saw_clock"
    assert "탁상시계" in fixed["lines"][0]["dialogue"] or "장식품" in fixed["lines"][0]["dialogue"]

    ts = await state.get_trial_state(sid)
    assert ts.current_counter_statement_id == "counter_yamano_saw_clock"


@pytest.mark.asyncio
async def test_request_hint_differs_by_stage_phase(episode):
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    testimony_hint = (await court.request_hint(sid))[0]
    assert testimony_hint["type"] == "helper_hint"
    assert testimony_hint["stage_phase"] == "testimony"
    assert "부검" in testimony_hint["hint"] or "신고 시각" in testimony_hint["hint"]

    await state.add_evidence(sid, "autopsy_report")
    await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순됩니다! 4시 사망인데 2시에 시체를 봤다고?",
        "objection",
        ["autopsy_report"],
    )

    ts = await state.get_trial_state(sid)
    assert ts.stage_phase == "cross_exam_free"

    cross_hint = (await court.request_hint(sid))[0]
    assert cross_hint["type"] == "helper_hint"
    assert cross_hint["stage_phase"] == "cross_exam_free"
    assert cross_hint["hint"] != testimony_hint["hint"]
    assert cross_hint["contradiction_index"] == 1
    assert "TV" in cross_hint["hint"] or "정전" in cross_hint["hint"] or "비디오" in cross_hint["hint"]


@pytest.mark.asyncio
async def test_repeated_hint_stays_on_current_contradiction(episode):
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    first = (await court.request_hint(sid))[0]
    second = (await court.request_hint(sid))[0]
    assert first["hint"] == second["hint"]
    assert first["contradiction_index"] == 0
    assert second["hint_level"] == 2


@pytest.mark.asyncio
async def test_hint_resets_after_contradiction_break(episode):
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await state.add_evidence(sid, "autopsy_report")
    await court.start_court(sid)

    await court.request_hint(sid)
    await court.request_hint(sid)

    await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순됩니다! 4시 사망인데 2시에 시체를 봤다고?",
        "objection",
        ["autopsy_report"],
    )

    cross_hint = (await court.request_hint(sid))[0]
    assert cross_hint["contradiction_index"] == 1
    assert cross_hint["hint_level"] == 1
    assert "TV" in cross_hint["hint"] or "정전" in cross_hint["hint"] or "비디오" in cross_hint["hint"]
    assert "장식품" not in cross_hint["hint"] and "시계" not in cross_hint["hint"]


@pytest.mark.asyncio
async def test_disaster_epitaph_witness_context_uses_isoeun():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM
    from backend.ai_services.judge_actor import JudgeActorLLM
    from backend.ai_services.prosecutor_actor import ProsecutorActorLLM
    from backend.ai_services.witness_actor import WitnessActorLLM
    from backend.core.free_dialogue_engine import FreeDialogueEngine

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_club")
    state = StateManager()
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    ts = await state.get_trial_state(sid)

    engine = FreeDialogueEngine(
        state,
        AnswerEvaluatorLLM(None),
        WitnessActorLLM(None),
        ProsecutorActorLLM(None),
        JudgeActorLLM(None),
    )
    context = await engine._build_free_dialogue_context(
        sid, episode, stage, "wit_ep_001", ts
    )

    assert context["character"]["name"] == "이소은"
    assert context["character"]["id"] == "wit_ep_001"
    assert stage.active_witness_id == "wit_ep_001"
    assert all("김탈북" not in lie for lie in context["allowed_lies"])
    isoeun_lies = [lie for lie in context["allowed_lies"] if "이소은" in lie]
    assert len(isoeun_lies) == 2
    prosecution = context["stage_context"]["prosecution_context"]
    assert "이소은" in prosecution["purpose"]
    assert "김탈북" not in prosecution["purpose"]
    assert context["case_summary"]["victim"] == "양진혁 (YJ그룹 공동창업자)"


@pytest.mark.asyncio
async def test_disaster_epitaph_trial2_hint_matches_current_round():
    from backend.services.episode_loader import load_episode

    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court.start_trial(sid, "trial_epitaph_2")

    ts = await state.get_trial_state(sid)
    ts.stage_hint_levels["stage_epitaph_car:testimony"] = 2
    await state.save_trial_state(sid, ts)

    hint = (await court.request_hint(sid))[0]
    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_car")
    assert hint["contradiction_index"] == 0
    assert hint["hint"] == stage.hints[0]
    assert "CCTV" in hint["hint"]
    assert "3차 재판" not in hint["hint"]


@pytest.mark.asyncio
async def test_disaster_epitaph_trial2_model_answers_under_100_chars():
    from backend.core.helper import Helper

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_car")
    helper = Helper()
    answers = stage.prosecution_context["model_answers"]
    assert answers == [EPITAPH_TRIAL2_MODEL_1, EPITAPH_TRIAL2_MODEL_2, EPITAPH_TRIAL2_MODEL_3]
    for index, answer in enumerate(answers):
        assert len(answer) <= 100
        assert helper.get_model_answer(stage, index) == answer


@pytest.mark.asyncio
async def test_disaster_epitaph_trial2_opening_submits_prosecutor_evidence():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    events = await court.start_court(sid, trial_id="trial_epitaph_2")
    types = [e["type"] for e in events]
    assert "evidence_submitted" in types
    submitted = [e["evidence_id"] for e in events if e["type"] == "evidence_submitted"]
    assert submitted[:2] == ["ev_ep_server_log", "ev_ep_laptop"]
    prosecutor_submit_idx = next(
        i for i, e in enumerate(events) if e.get("type") == "prosecutor_response"
    )
    first_evidence_idx = next(
        i for i, e in enumerate(events) if e.get("type") == "evidence_submitted"
    )
    first_actor_idx = next(i for i, e in enumerate(events) if e.get("type") == "actor_lines")
    stage_started_idx = next(i for i, e in enumerate(events) if e.get("type") == "stage_started")
    witness_idx = next(i for i, e in enumerate(events) if e.get("type") == "witness_testimony")
    witness_testimony = events[witness_idx]
    assert witness_testimony["is_fixed"] is True
    assert witness_testimony["statement_id"] == "stmt_minsoo_hack_left"
    assert witness_testimony["text"] == EPITAPH_TRIAL2_STMT
    assert witness_testimony["lines"][0]["dialogue"] == EPITAPH_TRIAL2_STMT
    assert "연행 호송차" in witness_testimony["lines"][0]["dialogue"]
    assert "우회전" in witness_testimony["lines"][0]["dialogue"]
    assert "소호 차량" not in witness_testimony["lines"][0]["dialogue"]
    assert "좌회전" not in witness_testimony["lines"][0]["dialogue"]
    assert prosecutor_submit_idx < first_evidence_idx < first_actor_idx < stage_started_idx < witness_idx
    actor_line_events = [e for e in events if e.get("type") == "actor_lines"]
    assert len(actor_line_events) >= 2
    opening = actor_line_events[0]
    assert opening["lines"][0]["dialogue"] == EPITAPH_TRIAL2_OPENING_JUDGE
    post_denial = actor_line_events[1]
    assert post_denial["lines"][0]["speaker"] == "judge_001"
    assert post_denial["lines"][1]["speaker"] == "pros_001"
    assert "좌회전" in post_denial["lines"][1]["dialogue"]
    assert "노트북" in post_denial["lines"][1]["dialogue"]
    prosecutor_submit = events[prosecutor_submit_idx]
    assert "서버 로그" in prosecutor_submit["lines"][0]["dialogue"]
    post_testimony = next(
        e for e in events if e.get("intervention_type") == "post_testimony"
    )
    assert "연행 호송차" in post_testimony["lines"][0]["dialogue"]
    assert "소호 차량" not in post_testimony["lines"][0]["dialogue"]

    inv = await state.get_inventory(sid)
    assert "ev_ep_laptop" in inv
    assert "ev_ep_server_log" in inv
    assert "ev_ep_kakao" not in inv
    assert "ev_ep_minsoo_opinion" not in inv


@pytest.mark.asyncio
async def test_disaster_epitaph_trial2_skip_seed_excludes_prosecutor_only_evidence():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    episode = load_episode("disaster_epitaph")
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court._seed_court_inventory(sid, episode)
    await court._seed_trial_skip_state(sid, episode, "trial_epitaph_2")

    inv = await state.get_inventory(sid)
    for garage_id in (
        "ev_ep_cctv_car",
        "ev_ep_autodrive_log",
        "ev_ep_wiretap",
        "ev_ep_anthony_id",
    ):
        assert garage_id in inv
    for prosecutor_only in (
        "ev_ep_laptop",
        "ev_ep_server_log",
        "ev_ep_kakao",
        "ev_ep_minsoo_opinion",
    ):
        assert prosecutor_only not in inv

    sanitized = await court.free_dialogue_engine._sanitize_selected_evidence_ids(
        sid,
        ["ev_ep_cctv_car", "ev_ep_server_log", "ev_ep_laptop"],
    )
    assert sanitized == ["ev_ep_cctv_car"]


@pytest.mark.asyncio
async def test_disaster_epitaph_garage_collect_syncs_inventory():
    state = StateManager()
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    garage_objects = (
        ("inv_ep_wiretap", "ev_ep_wiretap"),
        ("inv_ep_anthony_id", "ev_ep_anthony_id"),
        ("inv_ep_cctv_car", "ev_ep_cctv_car"),
        ("inv_ep_autodrive_log", "ev_ep_autodrive_log"),
    )
    for object_id, evidence_id in garage_objects:
        inv = await state.add_evidence(sid, evidence_id)
        assert evidence_id in inv

    inv = await state.get_inventory(sid)
    for _, evidence_id in garage_objects:
        assert evidence_id in inv


@pytest.mark.asyncio
async def test_disaster_epitaph_trial2_battle1_objection_succeeds():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_car")
    statement = stage.fixed_testimony_chain[0]
    evaluator = AnswerEvaluatorLLM(None)

    result = await evaluator.evaluate_stage_argument(
        stage_type=stage.stage_type.value,
        current_stage=stage,
        current_statement=statement,
        user_text=EPITAPH_TRIAL2_BATTLE1_USER,
        selected_evidence_ids=["ev_ep_cctv_car", "ev_ep_server_log"],
        selected_evidence_details=[],
        court_records=[],
    )
    assert result.verdict == "success"

    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court.start_court(sid, trial_id="trial_epitaph_2")
    events = await court.process_free_dialogue(
        sid,
        "stage_epitaph_car",
        EPITAPH_TRIAL2_BATTLE1_USER,
        "objection",
        ["ev_ep_cctv_car", "ev_ep_server_log"],
    )
    assert any(e.get("type") == "evidence_submitted" and e.get("evidence_id") == "ev_ep_minsoo_opinion" for e in events)
    assert any(e.get("type") == "witness_reaction" for e in events)


@pytest.mark.asyncio
async def test_disaster_epitaph_trial2_battle2_objection_unlocks_battle3():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_car")
    counter = stage.counter_by_id("counter_minsoo_lidar")
    evaluator = AnswerEvaluatorLLM(None)

    result = await evaluator.evaluate_stage_argument(
        stage_type=stage.stage_type.value,
        current_stage=stage,
        current_statement=counter,
        user_text=EPITAPH_TRIAL2_BATTLE2_USER,
        selected_evidence_ids=["ev_ep_minsoo_opinion", "ev_ep_server_log"],
        selected_evidence_details=[],
        court_records=[],
    )
    assert result.verdict == "success"

    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court.start_court(sid, trial_id="trial_epitaph_2")
    await court.process_free_dialogue(
        sid,
        "stage_epitaph_car",
        EPITAPH_TRIAL2_BATTLE1_USER,
        "objection",
        ["ev_ep_cctv_car", "ev_ep_server_log"],
    )
    events = await court.process_free_dialogue(
        sid,
        "stage_epitaph_car",
        EPITAPH_TRIAL2_BATTLE2_USER,
        "objection",
        ["ev_ep_minsoo_opinion", "ev_ep_server_log"],
    )
    assert not any(e.get("type") == "stage_cleared" for e in events)
    assert any(e.get("type") == "evidence_submitted" and e.get("evidence_id") == "ev_ep_kakao" for e in events)
    assert any(
        e.get("type") == "witness_reaction"
        and e.get("statement_id") == "counter_kakao_motive"
        for e in events
    )


@pytest.mark.asyncio
async def test_disaster_epitaph_trial2_battle3_objection_succeeds_and_adjourns():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_car")
    counter = stage.counter_by_id("counter_kakao_motive")
    evaluator = AnswerEvaluatorLLM(None)

    result = await evaluator.evaluate_stage_argument(
        stage_type=stage.stage_type.value,
        current_stage=stage,
        current_statement=counter,
        user_text=EPITAPH_TRIAL2_BATTLE3_USER,
        selected_evidence_ids=["ev_ep_kakao"],
        selected_evidence_details=[],
        court_records=[],
    )
    assert result.verdict == "success"

    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court.start_court(sid, trial_id="trial_epitaph_2")
    await court.process_free_dialogue(
        sid,
        "stage_epitaph_car",
        EPITAPH_TRIAL2_BATTLE1_USER,
        "objection",
        ["ev_ep_cctv_car", "ev_ep_server_log"],
    )
    await court.process_free_dialogue(
        sid,
        "stage_epitaph_car",
        EPITAPH_TRIAL2_BATTLE2_USER,
        "objection",
        ["ev_ep_minsoo_opinion", "ev_ep_server_log"],
    )
    events = await court.process_free_dialogue(
        sid,
        "stage_epitaph_car",
        EPITAPH_TRIAL2_BATTLE3_USER,
        "objection",
        ["ev_ep_kakao"],
    )
    assert any(e.get("type") == "stage_cleared" for e in events)
    assert any(
        e.get("type") == "judge_comment" and e.get("event_type") == "trial_adjourned"
        for e in events
    )
    assert any(
        e.get("type") == "prosecutor_pressure"
        and e.get("intervention_type") == "trial_adjourn_request"
        for e in events
    )


@pytest.mark.asyncio
async def test_disaster_epitaph_trial1_hints_follow_battle_progression():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court.start_court(sid)

    hint1 = (await court.request_hint(sid))[0]
    assert hint1["contradiction_index"] == 0
    assert "50mg" in hint1["hint"] and "20mg" in hint1["hint"]

    await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_1,
        "objection",
        ["ev_ep_vx_info", "stmt_epitaph_isoeun_1"],
    )
    hint2 = (await court.request_hint(sid))[0]
    assert hint2["contradiction_index"] == 1
    assert "오른손" in hint2["hint"]

    await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_2,
        "question",
        [],
    )

    await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_3,
        "objection",
        ["stmt_epitaph_isoeun_1", "stmt_epitaph_isoeun_2"],
    )
    hint4 = (await court.request_hint(sid))[0]
    assert hint4["contradiction_index"] == 3
    assert "10mg" in hint4["hint"] and "증언 #4" in hint4["hint"]


def test_disaster_epitaph_model_answers_under_100_chars():
    from backend.core.helper import Helper

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_club")
    helper = Helper()
    answers = stage.prosecution_context["model_answers"]
    assert answers == [
        EPITAPH_MODEL_ANSWER_1,
        EPITAPH_MODEL_ANSWER_2,
        EPITAPH_MODEL_ANSWER_3,
        EPITAPH_MODEL_ANSWER_4,
    ]
    for index, answer in enumerate(answers):
        assert len(answer) <= 100
        assert helper.get_model_answer(stage, index) == answer
        assert helper.get_model_answer_hint(stage, index).startswith("모범답안)")


@pytest.mark.asyncio
async def test_disaster_epitaph_start_court_at_trial_2_seeds_skip_state():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    events = await court.start_court(sid, trial_id="trial_epitaph_2")
    types = [e["type"] for e in events]
    assert "trial_started" in types
    assert not any(e.get("type") == "error" for e in events)
    assert next(e for e in events if e["type"] == "trial_started")["trial_id"] == "trial_epitaph_2"

    ts = await state.get_trial_state(sid)
    assert ts.current_trial_id == "trial_epitaph_2"
    assert ts.current_stage_id == "stage_epitaph_car"
    assert "trial_epitaph_1" in ts.cleared_trial_ids
    assert "stage_epitaph_club" in ts.cleared_stages

    inv = await state.get_inventory(sid)
    for evidence_id in (
        "ev_ep_cctv_car",
        "ev_ep_autodrive_log",
        "ev_ep_wiretap",
        "ev_ep_anthony_id",
        "ev_ep_laptop",
        "ev_ep_server_log",
    ):
        assert evidence_id in inv
    for trial1_only in (
        "ev_ep_autopsy",
        "ev_ep_medical",
        "ev_ep_vx_info",
        "ev_ep_glasses",
        "ev_ep_cctv_club",
        "ev_ep_club_flyer",
        "ev_ep_doctor_opinion",
    ):
        assert trial1_only not in inv
    for not_yet_submitted in ("ev_ep_kakao", "ev_ep_minsoo_opinion"):
        assert not_yet_submitted not in inv


@pytest.mark.asyncio
async def test_disaster_epitaph_start_court_at_trial_3_seeds_skip_state():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    events = await court.start_court(sid, trial_id="trial_epitaph_3")
    assert not any(e.get("type") == "error" for e in events)
    assert next(e for e in events if e["type"] == "trial_started")["trial_id"] == "trial_epitaph_3"

    ts = await state.get_trial_state(sid)
    assert ts.current_trial_id == "trial_epitaph_3"
    assert ts.current_stage_id == "stage_epitaph_final_stub"
    assert ts.cleared_trial_ids == ["trial_epitaph_1", "trial_epitaph_2"]
    assert "stage_epitaph_club" in ts.cleared_stages
    assert "stage_epitaph_car" in ts.cleared_stages


@pytest.mark.asyncio
async def test_disaster_epitaph_start_court_seeds_stage_and_inventory():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    events = await court.start_court(sid)
    types = [e["type"] for e in events]
    assert "trial_started" in types
    assert "stage_started" in types
    assert "witness_testimony" in types
    assert not any(e.get("type") == "error" for e in events)
    testimony = next(e for e in events if e["type"] == "witness_testimony")
    assert testimony["witness_id"] == "wit_ep_001"
    assert testimony["lines"][0]["dialogue"] == EPITAPH_FIXED_STMT_1
    ts = await state.get_trial_state(sid)
    assert ts.current_stage_id == "stage_epitaph_club"
    assert ts.stage_phase == "testimony"
    inv = await state.get_inventory(sid)
    assert "ev_ep_autopsy" in inv
    assert "ev_ep_vx_info" in inv
    assert "ev_ep_doctor_opinion" not in inv

    dialogue_events = await court.process_free_dialogue(
        sid,
        ts.current_stage_id,
        "증인님, 그날 밤 무슨 일이 있었나요?",
        "question",
        [],
    )
    assert not any(e.get("type") == "error" for e in dialogue_events)
    assert "orchestration_complete" in [e["type"] for e in dialogue_events]


@pytest.mark.asyncio
async def test_disaster_epitaph_battle1_fails_on_weak_logic_without_statement():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_club")
    statement = stage.fixed_testimony_chain[0]
    evaluator = AnswerEvaluatorLLM(None)

    result = await evaluator.evaluate_stage_argument(
        stage_type=stage.stage_type.value,
        current_stage=stage,
        current_statement=statement,
        user_text="술잔 20mg은 애매합니다.",
        selected_evidence_ids=["ev_ep_vx_info"],
        selected_evidence_details=[],
        court_records=[],
    )
    assert result.verdict == "fail"
    assert "모순" in result.reason or "용량" in result.reason or "VX" in result.reason or "갈래" in result.reason


@pytest.mark.asyncio
async def test_disaster_epitaph_battle1_nonsense_fails():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_club")
    statement = stage.fixed_testimony_chain[0]
    evaluator = AnswerEvaluatorLLM(None)

    result = await evaluator.evaluate_stage_argument(
        stage_type=stage.stage_type.value,
        current_stage=stage,
        current_statement=statement,
        user_text="안녕하세요, 날씨가 좋네요.",
        selected_evidence_ids=[],
        selected_evidence_details=[],
        court_records=[],
    )
    assert result.verdict == "fail"


@pytest.mark.asyncio
async def test_disaster_epitaph_battle1_partial_dose_only_fails():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_club")
    statement = stage.fixed_testimony_chain[0]
    evaluator = AnswerEvaluatorLLM(None)

    result = await evaluator.evaluate_stage_argument(
        stage_type=stage.stage_type.value,
        current_stage=stage,
        current_statement=statement,
        user_text="VX 50mg 20mg 술잔",
        selected_evidence_ids=["ev_ep_vx_info"],
        selected_evidence_details=[],
        court_records=[],
    )
    assert result.verdict == "fail"


@pytest.mark.asyncio
async def test_disaster_epitaph_battle1_objection_nonsense_costs_life():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court.start_court(sid)
    ts_before = await state.get_trial_state(sid)
    life_before = ts_before.stage_life

    events = await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        "그냥 이의 있습니다.",
        "objection",
        [],
    )
    assert any(e.get("type") == "life_update" and e.get("life_loss") == 1 for e in events)
    judge = next(e for e in events if e["type"] == "judge_intervention")
    assert judge["event_type"] == "objection_overruled"
    ts_after = await state.get_trial_state(sid)
    assert ts_after.stage_life == life_before - 1


@pytest.mark.asyncio
async def test_disaster_epitaph_battle3_partial_logic_fails():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_club")
    statement = stage.counter_by_id("stmt_epitaph_isoeun_2")
    evaluator = AnswerEvaluatorLLM(None)

    result = await evaluator.evaluate_stage_argument(
        stage_type=stage.stage_type.value,
        current_stage=stage,
        current_statement=statement,
        user_text="양진혁 손에 묻었다고 했죠?",
        selected_evidence_ids=[],
        selected_evidence_details=[],
        court_records=[],
    )
    assert result.verdict == "fail"


@pytest.mark.asyncio
async def test_disaster_epitaph_battle4_partial_logic_fails():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_club")
    stmt4 = stage.counter_by_id("stmt_epitaph_isoeun_4")
    evaluator = AnswerEvaluatorLLM(None)

    result = await evaluator.evaluate_stage_argument(
        stage_type=stage.stage_type.value,
        current_stage=stage,
        current_statement=stmt4,
        user_text="VX 피부 치사량 10mg입니다.",
        selected_evidence_ids=[],
        selected_evidence_details=[],
        court_records=[],
    )
    assert result.verdict == "fail"


@pytest.mark.asyncio
async def test_disaster_epitaph_battle2_nonsense_question_fails():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court.start_court(sid)
    await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_1,
        "objection",
        ["ev_ep_vx_info", "stmt_epitaph_isoeun_1"],
    )
    ts_before = await state.get_trial_state(sid)
    life_before = ts_before.stage_life

    events = await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        "오늘 날씨 어때요?",
        "question",
        [],
    )
    assert any(e.get("type") == "life_update" and e.get("life_loss") == 1 for e in events)
    assert any(e.get("type") == "battle2_interrogation_resolved" and not e.get("success") for e in events)
    ts_after = await state.get_trial_state(sid)
    assert ts_after.stage_life == life_before - 1


@pytest.mark.asyncio
async def test_disaster_epitaph_battle1_pass_without_exact_evidence():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_club")
    statement = stage.fixed_testimony_chain[0]
    evaluator = AnswerEvaluatorLLM(None)
    user_text = EPITAPH_MODEL_ANSWER_1

    result = await evaluator.evaluate_stage_argument(
        stage_type=stage.stage_type.value,
        current_stage=stage,
        current_statement=statement,
        user_text=user_text,
        selected_evidence_ids=[],
        selected_evidence_details=[],
        court_records=[],
    )
    assert result.verdict == "success"
    assert "용량" in result.reason or "증언 #1" in result.reason


@pytest.mark.asyncio
async def test_disaster_epitaph_battle1_pass_with_vx_and_statement1():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_club")
    statement = stage.fixed_testimony_chain[0]
    evaluator = AnswerEvaluatorLLM(None)

    result = await evaluator.evaluate_stage_argument(
        stage_type=stage.stage_type.value,
        current_stage=stage,
        current_statement=statement,
        user_text=EPITAPH_MODEL_ANSWER_1,
        selected_evidence_ids=["ev_ep_vx_info", "stmt_epitaph_isoeun_1"],
        selected_evidence_details=[],
        court_records=[],
    )
    assert result.verdict == "success"
    assert "용량" in result.reason or "증언 #1" in result.reason


@pytest.mark.asyncio
async def test_disaster_epitaph_battle1_success_submits_doctor_opinion():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court.start_court(sid)

    events = await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_1,
        "objection",
        ["ev_ep_vx_info", "stmt_epitaph_isoeun_1"],
    )
    types = [e["type"] for e in events]
    assert "phase_transition" in types
    phase = next(e for e in events if e["type"] == "phase_transition")
    assert phase["message"] == "증인에게 물어보고 싶은 것이 있습니다."
    assert "evidence_submitted" in types
    submitted = next(e for e in events if e["type"] == "evidence_submitted")
    assert submitted["evidence_id"] == "ev_ep_doctor_opinion"
    prosecutor = next(
        e
        for e in events
        if e["type"] == "prosecutor_response" and e.get("response_mode") == "fixed_submit"
    )
    assert prosecutor["lines"][0]["dialogue"] == EPITAPH_FIXED_PROSECUTOR_REBUTTAL
    inv = await state.get_inventory(sid)
    assert "ev_ep_doctor_opinion" in inv
    assert not any(
        e.get("type") == "witness_reaction"
        and e.get("statement_id") == "stmt_epitaph_isoeun_2"
        for e in events
    )
    intro = next(e for e in events if e["type"] == "actor_lines" and e.get("is_fixed"))
    assert intro["lines"][0]["dialogue"] == EPITAPH_FIXED_JUDGE_CROSS
    assert intro["lines"][1]["dialogue"] == EPITAPH_FIXED_DEFENSE_CROSS
    phase = next(e for e in events if e["type"] == "phase_transition")
    assert phase.get("battle2_interrogation") is True


@pytest.mark.asyncio
async def test_disaster_epitaph_battle2_question_skips_turn_scoring():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court.start_court(sid)
    await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_1,
        "objection",
        ["ev_ep_vx_info", "stmt_epitaph_isoeun_1"],
    )

    events = await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        "오른손에 VX가 왜 묻었는지 경위를 설명해 주세요.",
        "question",
        [],
    )
    types = [e["type"] for e in events]
    assert "witness_reaction" in types
    stmt2 = next(
        e
        for e in events
        if e["type"] == "witness_reaction" and e.get("statement_id") == "stmt_epitaph_isoeun_2"
    )
    assert stmt2["lines"][0]["dialogue"] == EPITAPH_FIXED_STMT_2
    assert "turn_contradiction_evaluated" not in types
    assert any(e.get("type") == "battle2_interrogation_resolved" and e.get("success") for e in events)


@pytest.mark.asyncio
async def test_disaster_epitaph_battle2_accepts_question_with_medical_evidence():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court.start_court(sid)
    await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_1,
        "objection",
        ["ev_ep_vx_info", "stmt_epitaph_isoeun_1"],
    )

    events = await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        "진료 기록에 나온 검출 결과, 왜 그런지 설명해 주세요.",
        "question",
        ["ev_ep_medical"],
    )
    types = [e["type"] for e in events]
    assert "error" not in types
    assert any(e.get("type") == "battle2_interrogation_resolved" and e.get("success") for e in events)


@pytest.mark.asyncio
async def test_disaster_epitaph_battle3_and_4_clear_stage():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court.start_court(sid)
    await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_1,
        "objection",
        ["ev_ep_vx_info", "stmt_epitaph_isoeun_1"],
    )
    await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_2,
        "question",
        [],
    )

    battle3 = await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_3,
        "objection",
        ["stmt_epitaph_isoeun_1", "stmt_epitaph_isoeun_2"],
    )
    types3 = [e["type"] for e in battle3]
    assert "witness_reaction" in types3
    stmt3 = next(
        e
        for e in battle3
        if e["type"] == "witness_reaction" and e.get("statement_id") == "stmt_epitaph_isoeun_3"
    )
    stmt4 = next(
        e
        for e in battle3
        if e["type"] == "witness_reaction" and e.get("statement_id") == "stmt_epitaph_isoeun_4"
    )
    assert stmt3["lines"][0]["dialogue"] == EPITAPH_FIXED_STMT_3
    assert stmt4["lines"][0]["dialogue"] == EPITAPH_FIXED_STMT_4

    battle4 = await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_4,
        "objection",
        ["ev_ep_vx_info", "stmt_epitaph_isoeun_4"],
    )
    types4 = [e["type"] for e in battle4]
    assert "stage_cleared" in types4
    breakdown = next(e for e in battle4 if e["type"] == "witness_breakdown")
    assert breakdown["lines"][0]["dialogue"] == EPITAPH_FIXED_CONFESSION


async def _advance_disaster_epitaph_to_battle4(court, sid):
    await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_1,
        "objection",
        ["ev_ep_vx_info", "stmt_epitaph_isoeun_1"],
    )
    await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_2,
        "question",
        [],
    )
    await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_3,
        "objection",
        ["stmt_epitaph_isoeun_1", "stmt_epitaph_isoeun_2"],
    )


@pytest.mark.asyncio
async def test_disaster_epitaph_battle3_pass_without_exact_evidence():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_club")
    statement = stage.counter_by_id("stmt_epitaph_isoeun_2")
    evaluator = AnswerEvaluatorLLM(None)
    user_text = EPITAPH_MODEL_ANSWER_3

    result = await evaluator.evaluate_stage_argument(
        stage_type=stage.stage_type.value,
        current_stage=stage,
        current_statement=statement,
        user_text=user_text,
        selected_evidence_ids=[],
        selected_evidence_details=[],
        court_records=[],
    )
    assert result.verdict == "success"
    assert "순서" in result.reason or "모순" in result.reason


@pytest.mark.asyncio
async def test_disaster_epitaph_battle4_pass_without_exact_evidence():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_club")
    stmt4 = stage.counter_by_id("stmt_epitaph_isoeun_4")
    evaluator = AnswerEvaluatorLLM(None)
    user_text = EPITAPH_MODEL_ANSWER_4

    result = await evaluator.evaluate_stage_argument(
        stage_type=stage.stage_type.value,
        current_stage=stage,
        current_statement=stmt4,
        user_text=user_text,
        selected_evidence_ids=[],
        selected_evidence_details=[],
        court_records=[],
    )
    assert result.verdict == "success"


@pytest.mark.asyncio
async def test_disaster_epitaph_battle4_success_with_minimal_text():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_club")
    stmt4 = stage.counter_by_id("stmt_epitaph_isoeun_4")
    evaluator = AnswerEvaluatorLLM(None)

    result = await evaluator.evaluate_stage_argument(
        stage_type=stage.stage_type.value,
        current_stage=stage,
        current_statement=stmt4,
        user_text=EPITAPH_MODEL_ANSWER_4,
        selected_evidence_ids=["ev_ep_vx_info", "stmt_epitaph_isoeun_4"],
        selected_evidence_details=[],
        court_records=[],
    )
    assert result.verdict == "success"

    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court.start_court(sid)
    await _advance_disaster_epitaph_to_battle4(court, sid)

    battle4 = await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        EPITAPH_MODEL_ANSWER_4,
        "objection",
        [],
    )
    assert "stage_cleared" in [e["type"] for e in battle4]
    breakdown = next(e for e in battle4 if e["type"] == "witness_breakdown")
    assert breakdown["lines"][0]["speaker"] == "wit_ep_001"
    assert breakdown["lines"][0]["dialogue"] == EPITAPH_FIXED_CONFESSION


@pytest.mark.asyncio
async def test_disaster_epitaph_battle4_full_success_sequence():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM

    episode = load_episode("disaster_epitaph")
    stage = episode.get_stage("stage_epitaph_club")
    stmt4 = stage.counter_by_id("stmt_epitaph_isoeun_4")
    evaluator = AnswerEvaluatorLLM(None)
    user_text = EPITAPH_MODEL_ANSWER_4

    result = await evaluator.evaluate_stage_argument(
        stage_type=stage.stage_type.value,
        current_stage=stage,
        current_statement=stmt4,
        user_text=user_text,
        selected_evidence_ids=["ev_ep_vx_info", "stmt_epitaph_isoeun_4"],
        selected_evidence_details=[],
        court_records=[],
    )
    assert result.verdict == "success"

    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court.start_court(sid)
    await _advance_disaster_epitaph_to_battle4(court, sid)

    battle4 = await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        user_text,
        "objection",
        ["ev_ep_vx_info", "stmt_epitaph_isoeun_4"],
    )
    types = [e["type"] for e in battle4]
    assert "stage_cleared" in types
    assert "witness_breakdown" in types
    assert "prosecutor_pressure" in types
    assert "judge_comment" in types
    assert not any(
        e.get("type") == "witness_reaction" and e.get("source") == "objection_fail" for e in battle4
    )

    breakdown_idx = types.index("witness_breakdown")
    prosecutor_idx = types.index("prosecutor_pressure")
    judge_idx = types.index("judge_comment")
    cleared_idx = types.index("stage_cleared")
    assert breakdown_idx < prosecutor_idx < judge_idx < cleared_idx

    breakdown = next(e for e in battle4 if e["type"] == "witness_breakdown")
    prosecutor = next(e for e in battle4 if e["type"] == "prosecutor_pressure")
    judge = next(e for e in battle4 if e["type"] == "judge_comment" and e.get("event_type") == "trial_adjourned")

    assert breakdown["is_fixed"] is True
    assert breakdown["lines"][0]["speaker"] == "wit_ep_001"
    assert breakdown["lines"][0]["dialogue"] == EPITAPH_FIXED_CONFESSION
    assert prosecutor["is_fixed"] is True
    assert prosecutor["lines"][0]["speaker"] == "pros_001"
    assert prosecutor["lines"][0]["dialogue"] == EPITAPH_FIXED_PROSECUTOR_ADJOURN
    assert judge["is_fixed"] is True
    assert judge["lines"][0]["speaker"] == "judge_001"
    assert judge["lines"][0]["dialogue"] == EPITAPH_FIXED_JUDGE_ADJOURN


@pytest.mark.asyncio
async def test_disaster_epitaph_objection_failure_separates_judge_and_witness():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")
    await court.start_court(sid)
    await _advance_disaster_epitaph_to_battle4(court, sid)

    events = await court.process_free_dialogue(
        sid,
        "stage_epitaph_club",
        "그건 아닌 것 같습니다.",
        "objection",
        ["ev_ep_vx_info", "stmt_epitaph_isoeun_4"],
    )
    judge = next(e for e in events if e["type"] == "judge_intervention")
    witness = next(
        e for e in events if e["type"] == "witness_reaction" and e.get("source") == "objection_fail"
    )
    assert judge["lines"][0]["speaker"] == "judge_001"
    assert witness["lines"][0]["speaker"] == "wit_ep_001"
    assert "저는" not in judge["lines"][0]["dialogue"]


@pytest.mark.asyncio
async def test_judge_actor_filters_witness_voice():
    from backend.ai_services.judge_actor import JudgeActorLLM
    from backend.schemas.court import ActorLine, ActorResponse
    from backend.schemas.trial import AnswerVerdict, DefenseArgumentEvaluation, RelevanceLevel

    actor = JudgeActorLLM(None)
    evaluation = DefenseArgumentEvaluation(
        relevance=RelevanceLevel.PARTIALLY_RELEVANT,
        core_match_score=0.2,
        logic_score=0.2,
        evidence_usage_score=0.3,
        verdict=AnswerVerdict.FAIL,
        reason="아직 부족합니다.",
    )
    response = ActorResponse(
        lines=[
            ActorLine(
                speaker="judge_001",
                dialogue="저는 제 손에 닿은 양이 치사량보다 훨씬 적었다고 생각합니다.",
                animation_tag="think",
            )
        ]
    )
    sanitized = actor._sanitize(
        response,
        "objection_overruled",
        evaluation,
        None,
        ["ev_ep_vx_info"],
    )
    assert sanitized.lines[0].speaker == "judge_001"
    assert "저는" not in sanitized.lines[0].dialogue
    assert "기각" in sanitized.lines[0].dialogue or "쟁점" in sanitized.lines[0].dialogue

    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("disaster_epitaph", difficulty="easy")

    events = await court.start_trial(sid, "trial_epitaph_1")
    opening = next(e for e in events if e["type"] == "actor_lines")
    lines = opening["lines"]
    assert lines[0]["dialogue"] == EPITAPH_FIXED_OPENING_JUDGE
    assert lines[1]["dialogue"] == EPITAPH_FIXED_OPENING_DEFENDANT


@pytest.mark.asyncio
async def test_first_contradiction_break_emits_contradiction_helper(episode):
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await state.add_evidence(sid, "autopsy_report")
    await court.start_court(sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순됩니다! 4시 사망인데 2시에 시체를 봤다고?",
        "objection",
        ["autopsy_report"],
    )
    types = [e["type"] for e in events]
    transition = next(e for e in events if e["type"] == "phase_transition")
    assert transition["to_phase"] == "cross_exam_free"
    assert "helper_lines" not in transition
    assert "helper_success_cheer" not in types

    helper = next(e for e in events if e["type"] == "contradiction_helper")
    stage = episode.get_stage(STAGE_ID)
    assert helper["helper_lines"] == stage.contradiction_helper_lines[0]
    assert helper["broken_index"] == 0
    assert types.index("witness_reaction") < types.index("contradiction_helper")


@pytest.mark.asyncio
async def test_second_contradiction_break_emits_contradiction_helper(episode):
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await state.add_evidence(sid, "autopsy_report")
    await state.add_evidence(sid, "outage_record")
    await court.start_court(sid)

    await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순됩니다! 4시 사망인데 2시에 시체를 봤다고?",
        "objection",
        ["autopsy_report"],
    )

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "정전 기록에 따르면 TV나 비디오는 작동할 수 없습니다! 정전 중 모순입니다!",
        "objection",
        ["outage_record"],
    )
    types = [e["type"] for e in events]
    assert "helper_success_cheer" not in types
    helper = next(e for e in events if e["type"] == "contradiction_helper")
    stage = episode.get_stage(STAGE_ID)
    assert helper["helper_lines"] == stage.contradiction_helper_lines[1]
    assert helper["broken_index"] == 1


@pytest.mark.asyncio
async def test_objection_fail_does_not_emit_success_cheer():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "그냥 이상해요.",
        "objection",
        [],
    )
    types = [e["type"] for e in events]
    assert "helper_success_cheer" not in types
    assert "contradiction_helper" not in types


async def _enter_cross_exam_free(court, sid):
    await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순됩니다! 4시 사망인데 2시에 시체를 봤다고?",
        "objection",
        ["autopsy_report"],
    )


@pytest.mark.asyncio
async def test_turn_contradiction_severe_emits_judge_intervention():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await state.add_evidence(sid, "autopsy_report")
    await court.start_court(sid)
    await _enter_cross_exam_free(court, sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순되지 않습니까? 증거와 맞지 않습니다!",
        "question",
    )
    types = [e["type"] for e in events]
    planned = next(e for e in events if e["type"] == "orchestration_planned")
    assert planned["responders"][-1] == "judge"
    assert "turn_contradiction_evaluated" not in types
    assert "defense_argument_evaluated" in types
    assert "judge_intervention" in types
    judge = next(e for e in events if e["type"] == "judge_intervention")
    assert judge["lines"][0]["speaker"] == "judge_001"
    assert judge["trigger"] == "decisive"
    witness_idx = next(i for i, e in enumerate(events) if e["type"] == "witness_reaction")
    judge_idx = next(i for i, e in enumerate(events) if e["type"] == "judge_intervention")
    assert witness_idx < judge_idx


@pytest.mark.asyncio
async def test_objection_fail_deducts_life():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)
    ts = await state.get_trial_state(sid)
    assert ts.stage_life == 5

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "그냥 이상해요.",
        "objection",
        [],
    )
    types = [e["type"] for e in events]
    assert "judge_intervention" in types
    assert "life_update" in types
    life = next(e for e in events if e["type"] == "life_update")
    assert life["life_loss"] == 1
    assert life["remaining_life"] == 4
    ts_after = await state.get_trial_state(sid)
    assert ts_after.stage_life == 4


@pytest.mark.asyncio
async def test_decisive_fail_testimony_deducts_life():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순됩니다! 완전히 틀렸습니다!",
        "question",
    )
    types = [e["type"] for e in events]
    assert "judge_intervention" in types
    assert "life_update" in types
    judge = next(e for e in events if e["type"] == "judge_intervention")
    assert judge["trigger"] == "decisive"
    life = next(e for e in events if e["type"] == "life_update")
    assert life["life_loss"] == 1
    ts_after = await state.get_trial_state(sid)
    assert ts_after.stage_life == 4


@pytest.mark.asyncio
async def test_judge_intervention_life_zero_emits_stage_failed():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)
    ts = await state.get_trial_state(sid)
    ts.stage_life = 1
    await state.save_trial_state(sid, ts)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "그냥 이상해요.",
        "objection",
        [],
    )
    types = [e["type"] for e in events]
    assert "life_update" in types
    assert "stage_failed" in types
    life = next(e for e in events if e["type"] == "life_update")
    assert life["remaining_life"] == 0
    ts_after = await state.get_trial_state(sid)
    assert ts_after.stage_life == 0
    assert ts_after.failed_stage_id == STAGE_ID


@pytest.mark.asyncio
async def test_turn_contradiction_none_skips_judge_intervention():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "그때 날씨는 어땠나요?",
        "question",
    )
    types = [e["type"] for e in events]
    assert "turn_contradiction_evaluated" not in types
    assert "judge_intervention" not in types
    assert "orchestration_complete" in types


@pytest.mark.asyncio
async def test_cross_exam_evidence_question_coerces_objection_judge():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await state.add_evidence(sid, "autopsy_report")
    await state.add_evidence(sid, "outage_record")
    await court.start_court(sid)
    await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순됩니다! 4시 사망인데 2시에 시체를 봤다고?",
        "objection",
        ["autopsy_report"],
    )

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "정전 기록에 따르면 TV나 비디오는 작동할 수 없습니다!",
        "question",
        ["outage_record"],
    )
    types = [e["type"] for e in events]
    planned = next(e for e in events if e["type"] == "orchestration_planned")
    assert planned["mode"] == "objection"
    assert planned["responders"] == ["judge"]
    assert "judge_intervention" in types
    judge = next(e for e in events if e["type"] == "judge_intervention")
    assert judge["lines"][0]["speaker"] == "judge_001"


@pytest.mark.asyncio
async def test_decisive_cross_exam_judge_after_rebuttal_chain():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await state.add_evidence(sid, "autopsy_report")
    await court.start_court(sid)
    await _enter_cross_exam_free(court, sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "부검 기록과 증언이 모순되지 않습니까?",
        "question",
    )
    types = [e["type"] for e in events]
    planned = next(e for e in events if e["type"] == "orchestration_planned")
    assert planned["responders"][-1] == "judge"
    witness_idx = next(i for i, e in enumerate(events) if e["type"] == "witness_reaction")
    judge_idx = next(i for i, e in enumerate(events) if e["type"] == "judge_intervention")
    assert witness_idx < judge_idx
    judge = events[judge_idx]
    assert judge["lines"][0]["speaker"] == "judge_001"


@pytest.mark.asyncio
async def test_turn_contradiction_mock_none_in_cross_exam():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await state.add_evidence(sid, "autopsy_report")
    await court.start_court(sid)
    await _enter_cross_exam_free(court, sid)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "증인, 그때 정말 2시였습니까?",
        "question",
    )
    types = [e["type"] for e in events]
    assert "turn_contradiction_evaluated" in types
    evaluated = next(e for e in events if e["type"] == "turn_contradiction_evaluated")
    assert evaluated["evaluation"]["severity"] in ("none", "minor")
    assert evaluated["evaluation"]["intervention_needed"] is False
    assert "judge_intervention" not in types
    assert "orchestration_complete" in types


@pytest.mark.asyncio
async def test_turn_contradiction_context_includes_batch_lines():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM
    from backend.ai_services.judge_actor import JudgeActorLLM
    from backend.ai_services.prosecutor_actor import ProsecutorActorLLM
    from backend.ai_services.witness_actor import WitnessActorLLM
    from backend.core.free_dialogue_engine import FreeDialogueEngine
    from backend.schemas.trial import ContradictionSeverity, TurnContradictionEvaluation

    state = StateManager()
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    ts = await state.get_trial_state(sid)
    ts.stage_phase = "cross_exam_free"
    await state.save_trial_state(sid, ts)
    episode = load_episode("turnabout_clock")
    stage = episode.get_stage(STAGE_ID)

    captured: dict = {}

    async def fake_evaluate(**kwargs):
        captured.update(kwargs)
        evaluation = TurnContradictionEvaluation(
            intervention_needed=False,
            severity=ContradictionSeverity.NONE,
            reason="mock",
        )
        from backend.schemas.court import ActorResponse

        return evaluation, ActorResponse(lines=[])

    judge = JudgeActorLLM(None)
    judge.evaluate_turn_contradiction = fake_evaluate
    engine = FreeDialogueEngine(
        state,
        AnswerEvaluatorLLM(None),
        WitnessActorLLM(None),
        ProsecutorActorLLM(None),
        judge,
    )

    batch = [
        {"speaker": "wit_001", "text": "증인 답변"},
        {"speaker": "pros_001", "text": "검사 반박"},
    ]
    events = await engine._evaluate_turn_contradiction(
        sid,
        episode,
        stage,
        "변호인 질문",
        "question",
        batch,
        [],
        judge_trigger="none",
    )
    assert captured["user_text"] == "변호인 질문"
    assert captured["turn_batch_lines"] == batch
    assert events[0]["type"] == "turn_contradiction_evaluated"
    assert events[0]["turn_batch_lines"] == batch


@pytest.mark.asyncio
async def test_turn_contradiction_mock_severe_via_monkeypatch():
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM
    from backend.ai_services.judge_actor import JudgeActorLLM
    from backend.ai_services.prosecutor_actor import ProsecutorActorLLM
    from backend.ai_services.witness_actor import WitnessActorLLM
    from backend.core.free_dialogue_engine import FreeDialogueEngine
    from backend.schemas.court import ActorLine, ActorResponse
    from backend.schemas.trial import AnswerVerdict, ContradictionSeverity, TurnContradictionEvaluation

    state = StateManager()
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    ts = await state.get_trial_state(sid)
    ts.stage_phase = "cross_exam_free"
    ts.stage_life = 3
    await state.save_trial_state(sid, ts)
    episode = load_episode("turnabout_clock")
    stage = episode.get_stage(STAGE_ID)

    async def fake_severe(**kwargs):
        evaluation = TurnContradictionEvaluation(
            intervention_needed=True,
            severity=ContradictionSeverity.SEVERE,
            verdict=AnswerVerdict.FAIL,
            reason="논리 붕괴",
            life_loss=1,
        )
        response = ActorResponse(
            lines=[ActorLine(speaker="judge_001", dialogue="변호인, 그 주장은 받아들일 수 없습니다.", animation_tag="think")]
        )
        return evaluation, response

    judge = JudgeActorLLM(None)
    judge.evaluate_turn_contradiction = fake_severe
    engine = FreeDialogueEngine(
        state,
        AnswerEvaluatorLLM(None),
        WitnessActorLLM(None),
        ProsecutorActorLLM(None),
        judge,
    )

    events = await engine._evaluate_turn_contradiction(
        sid,
        episode,
        stage,
        "모순입니다!",
        "question",
        [{"speaker": "pros_001", "text": "성립하지 않습니다"}],
        [],
    )
    types = [e["type"] for e in events]
    assert "judge_intervention" in types
    assert "life_update" in types
    life = next(e for e in events if e["type"] == "life_update")
    assert life["life_loss"] == 1


@pytest.mark.asyncio
async def test_turn_contradiction_intervention_zero_life_loss_still_decrements():
    """LLM may return intervention_needed=true but life_loss=0 — engine must still penalize."""
    from backend.ai_services.answer_evaluator import AnswerEvaluatorLLM
    from backend.ai_services.judge_actor import JudgeActorLLM
    from backend.ai_services.prosecutor_actor import ProsecutorActorLLM
    from backend.ai_services.witness_actor import WitnessActorLLM
    from backend.core.free_dialogue_engine import FreeDialogueEngine
    from backend.schemas.court import ActorLine, ActorResponse
    from backend.schemas.trial import AnswerVerdict, ContradictionSeverity, TurnContradictionEvaluation

    state = StateManager()
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    ts = await state.get_trial_state(sid)
    ts.stage_phase = "cross_exam_free"
    ts.stage_life = 3
    await state.save_trial_state(sid, ts)
    episode = load_episode("turnabout_clock")
    stage = episode.get_stage(STAGE_ID)

    async def fake_intervention_no_life(**kwargs):
        evaluation = TurnContradictionEvaluation(
            intervention_needed=True,
            severity=ContradictionSeverity.SEVERE,
            verdict=AnswerVerdict.FAIL,
            reason="터무니없는 주장",
            life_loss=0,
        )
        response = ActorResponse(
            lines=[ActorLine(speaker="judge_001", dialogue="변호인, 그 주장은 받아들일 수 없습니다.", animation_tag="think")]
        )
        return evaluation, response

    judge = JudgeActorLLM(None)
    judge.evaluate_turn_contradiction = fake_intervention_no_life
    engine = FreeDialogueEngine(
        state,
        AnswerEvaluatorLLM(None),
        WitnessActorLLM(None),
        ProsecutorActorLLM(None),
        judge,
    )

    events = await engine._evaluate_turn_contradiction(
        sid,
        episode,
        stage,
        "외계인이 범인입니다",
        "question",
        [{"speaker": "wit_001", "text": "그건 터무니없는 추측입니다"}],
        [],
    )
    types = [e["type"] for e in events]
    assert "judge_intervention" in types
    assert "life_update" in types
    life = next(e for e in events if e["type"] == "life_update")
    assert life["life_loss"] == 1
    assert life["remaining_life"] == 2
    ts_after = await state.get_trial_state(sid)
    assert ts_after.stage_life == 2


def test_sanitize_character_names():
    from backend.core.korean_name_sanitizer import sanitize_character_names

    assert sanitize_character_names("여러분, 바로 이 소은 증인은") == "여러분, 바로 이소은 증인은"
    assert sanitize_character_names("양 진혁가 쓰러졌다") == "양진혁가 쓰러졌다"
    assert sanitize_character_names("임 민수 증인") == "임민수 증인"
    assert sanitize_character_names("앤 서니가") == "앤서니가"
    assert sanitize_character_names("Anthony가") == "앤서니가"
    assert sanitize_character_names("이소은은 정상") == "이소은은 정상"
    assert (
        sanitize_character_names("앤서니가 의도적으로 소호 차량을 조종했다")
        == "앤서니가 의도적으로 연행 호송차를 조종했다"
    )
    assert sanitize_character_names("소호의 차량 해킹") == "연행 호송차 해킹"
    assert sanitize_character_names("소호차량 조종") == "연행 호송차 조종"


def test_prosecutor_sanitize_fixes_spaced_witness_name():
    from backend.ai_services.prosecutor_actor import ProsecutorActorLLM
    from backend.schemas.court import ActorLine, ActorResponse
    from backend.schemas.trial import ProsecutorPlanMode

    actor = ProsecutorActorLLM(None)
    response = ActorResponse(
        lines=[ActorLine(speaker="pros_001", dialogue="바로 이 소은 증인은 결정적입니다.", animation_tag="basic")]
    )
    sanitized = actor._sanitize(response, ProsecutorPlanMode.PRESSURE)
    assert sanitized.lines[0].dialogue == "바로 이소은 증인은 결정적입니다."


def test_sanitize_turn_intervention_forces_life_loss():
    from backend.ai_services.judge_actor import JudgeActorLLM, TurnContradictionLLMResult
    from backend.schemas.trial import AnswerVerdict, ContradictionSeverity

    judge = JudgeActorLLM(None)
    result = TurnContradictionLLMResult(
        intervention_needed=True,
        severity=ContradictionSeverity.SEVERE,
        verdict=AnswerVerdict.FAIL,
        reason="논리 붕괴",
        life_loss=0,
    )
    evaluation, response = judge._sanitize_turn_result(result, remaining_life=3)
    assert evaluation.intervention_needed is True
    assert evaluation.life_loss == 1
    assert response.lines


@pytest.mark.asyncio
async def test_cross_exam_nonsense_statement_triggers_life_loss():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)
    ts = await state.get_trial_state(sid)
    ts.stage_phase = "cross_exam_free"
    ts.stage_life = 3
    await state.save_trial_state(sid, ts)

    events = await court.process_free_dialogue(
        sid,
        STAGE_ID,
        "증인, 범인은 외계인이 틀림없습니다!",
        "question",
    )
    types = [e["type"] for e in events]
    assert "judge_intervention" in types
    assert "life_update" in types
    life = next(e for e in events if e["type"] == "life_update")
    assert life["life_loss"] == 1
    ts_after = await state.get_trial_state(sid)
    assert ts_after.stage_life == 2


def test_helper_contradiction_chain_index(episode):
    from backend.core.helper import HELPER_SUCCESS_LINES, Helper

    helper = Helper()
    stage = episode.get_stage(STAGE_ID)
    chain = helper.build_statement_chain(stage)
    assert chain == [
        "stmt_yamano_reported_at_two",
        "counter_yamano_tv_sound",
        "counter_yamano_saw_clock",
        "counter_yamano_prove_delay",
    ]
    assert helper.get_success_cheer_lines() == HELPER_SUCCESS_LINES
    assert helper.get_helper_lines_after_break(stage, 0) is not None
    assert helper.get_helper_lines_after_break(stage, 1) is not None


@pytest.mark.asyncio
async def test_response_verifier_rejects_trial2_soho_vehicle_mislabel():
    verifier = ResponseVerifierLLM(None)
    response = ActorResponse(
        lines=[
            ActorLine(
                speaker="pros_001",
                dialogue=(
                    "이 증언이야말로 앤서니가 의도적으로 소호 차량을 조종했다는 "
                    "검찰의 주장을 확실히 입증합니다."
                ),
                animation_tag="basic",
            )
        ]
    )
    result = await verifier.verify(
        role="prosecutor",
        stage_id="stage_epitaph_car",
        stage_phase="cross_exam_free",
        response=response,
        user_text="소호 차량이 아니라 경찰차야",
        current_statement={"text": "앤서니가 연행 호송차에 우회전 명령을 보냈습니다."},
        turn_batch_lines=[],
        forbidden_claims=[],
        inventory_evidence=[],
    )
    assert result.valid is False
    assert any("소호" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_response_verifier_rejects_prosecutor_undermining():
    verifier = ResponseVerifierLLM(None)
    response = ActorResponse(
        lines=[
            ActorLine(
                speaker="pros_001",
                dialogue="증인의 말은 믿기 어렵습니다. 증언이 틀렸을 수도 있습니다.",
                animation_tag="basic",
            )
        ]
    )
    result = await verifier.verify(
        role="prosecutor",
        stage_id="stage_epitaph_car",
        stage_phase="cross_exam_free",
        response=response,
        user_text="증인 말이 맞습니까?",
        current_statement={"text": "앤서니가 우회전 명령을 보냈다."},
        turn_batch_lines=[{"speaker": "wit_ep_002", "text": "앤서니가 우회전 명령을 보냈습니다."}],
        forbidden_claims=[],
        inventory_evidence=[],
    )
    assert result.valid is False
    assert any("약화" in issue or "부정" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_response_verifier_rejects_trial2_witness_axis_flip():
    verifier = ResponseVerifierLLM(None)
    response = ActorResponse(
        lines=[
            ActorLine(
                speaker="wit_ep_002",
                dialogue="아닙니다, 사실 차량은 좌회전이 맞습니다.",
                animation_tag="basic",
            )
        ]
    )
    result = await verifier.verify(
        role="witness",
        stage_id="stage_epitaph_car",
        stage_phase="cross_exam_free",
        response=response,
        user_text="정말 우회전이었나요?",
        current_statement={"text": "앤서니가 급가속 후 우회전 명령을 보냈습니다."},
        turn_batch_lines=[],
        forbidden_claims=[],
        inventory_evidence=[],
    )
    assert result.valid is False
    assert any("핵심 주장" in issue for issue in result.issues)
