import pytest
from pydantic import ValidationError

from backend.core.court_orchestrator import CourtOrchestrator
from backend.core.state_manager import StateManager
from backend.schemas.trial import DefenseArgumentPayload

STAGE_ID = "stage_yamano_chain"


@pytest.mark.asyncio
async def test_defense_argument_payload_limits():
    with pytest.raises(ValidationError):
        DefenseArgumentPayload(
            session_id="s",
            stage_id="stage",
            text="정상 주장",
            selected_evidence_ids=["autopsy_report", "outage_record", "passport"],
        )

    with pytest.raises(ValidationError):
        DefenseArgumentPayload(
            session_id="s",
            stage_id="stage",
            text="가" * 101,
            selected_evidence_ids=[],
        )


@pytest.mark.asyncio
async def test_easy_and_hard_helper_flags():
    easy_state = StateManager()
    easy_sid = await easy_state.create_session("turnabout_clock", difficulty="easy")
    easy_court = CourtOrchestrator(easy_state, api_key=None)
    await easy_court.start_court(easy_sid)
    assert (await easy_state.get_trial_state(easy_sid)).helper_enabled is True
    assert (await easy_court.request_hint(easy_sid))[0]["type"] == "helper_hint"

    hard_state = StateManager()
    hard_sid = await hard_state.create_session("turnabout_clock", difficulty="hard")
    hard_court = CourtOrchestrator(hard_state, api_key=None)
    await hard_court.start_court(hard_sid)
    assert (await hard_state.get_trial_state(hard_sid)).helper_enabled is False
    assert (await hard_court.request_hint(hard_sid))[0]["type"] == "error"


@pytest.mark.asyncio
async def test_vs_witness_success_adds_counter_record_and_damage():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await state.add_evidence(sid, "autopsy_report")
    await court.start_court(sid)

    events = await court.process_defense_argument(
        sid,
        STAGE_ID,
        "부검 기록에 따르면 사망은 4시부터 5시인데 증인은 2시에 시체를 발견했다고 합니다.",
        ["autopsy_report"],
    )
    ts = await state.get_trial_state(sid)
    assert ts.witness_mental_by_stage[STAGE_ID] == 75
    assert any(e["type"] == "witness_counter" for e in events)
    assert any(e["type"] == "judge_comment" for e in events)
    assert any(e["type"] in ("witness_reaction", "witness_shaken") for e in events)

    records = await state.get_court_records(sid)
    counter = next(r for r in records if r.statement_id == "counter_yamano_tv_sound")
    assert counter.usable_as_evidence is True
    assert counter.source == "witness_counter"
    assert "counter_yamano_tv_sound" in ts.usable_statement_evidence_ids


@pytest.mark.asyncio
async def test_basic_fail_uses_judge_without_default_prosecutor_pressure():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    events = await court.process_defense_argument(
        sid,
        STAGE_ID,
        "그냥 아닌 것 같습니다.",
        ["autopsy_report"],
    )
    types = [e["type"] for e in events]
    assert "judge_comment" in types
    assert "life_update" in types
    assert "witness_reaction" in types
    assert "prosecutor_pressure" not in types


@pytest.mark.asyncio
async def test_no_evidence_fail_gets_conditional_prosecutor_pressure():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    await court.start_court(sid)

    events = await court.process_defense_argument(
        sid,
        STAGE_ID,
        "증인의 말은 그냥 의심스럽습니다.",
        [],
    )
    types = [e["type"] for e in events]
    assert "judge_comment" in types
    assert "prosecutor_pressure" in types


@pytest.mark.asyncio
async def test_vs_witness_fail_loses_life_and_can_fail_stage():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="hard")
    await court.start_court(sid)

    events = []
    for _ in range(3):
        events = await court.process_defense_argument(
            sid,
            STAGE_ID,
            "그냥 아닌 것 같습니다.",
            [],
        )

    ts = await state.get_trial_state(sid)
    assert ts.stage_life == 0
    assert ts.failed_stage_id == STAGE_ID
    assert any(e["type"] == "stage_failed" for e in events)


@pytest.mark.asyncio
async def test_mock_stage_clear_and_scores_roll_up():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", difficulty="easy")
    for eid in ["autopsy_report", "outage_record", "thinker_clock", "passport"]:
        await state.add_evidence(sid, eid)
    await court.start_court(sid)

    answers = [
        (
            "부검 기록에 따르면 사망은 4시부터 5시인데 증인은 2시에 시체를 발견했다고 합니다.",
            ["autopsy_report"],
        ),
        (
            "정전 기록상 TV는 작동할 수 없으니 TV 소리로 2시를 알았다는 말은 모순입니다.",
            ["outage_record"],
        ),
        (
            "생각하는 사람 장식품은 겉보기엔 시계처럼 보이지 않아 직접 본 것이 아닙니다.",
            ["thinker_clock"],
        ),
        (
            "여권에 따르면 뉴욕 귀국으로 시계가 2시간 늦어져 있었고 이것이 모순을 설명합니다.",
            ["passport"],
        ),
    ]

    events = []
    for text, evidence_ids in answers:
        events = await court.process_defense_argument(sid, STAGE_ID, text, evidence_ids)

    ts = await state.get_trial_state(sid)
    assert STAGE_ID in ts.cleared_stages
    assert ts.stage_scores[STAGE_ID] > 0
    assert any(e["type"] == "stage_cleared" for e in events)
    assert any(e["type"] in ("ending", "episode_score") for e in events)
