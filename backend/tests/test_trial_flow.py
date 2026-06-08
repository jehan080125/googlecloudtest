import pytest

from backend.ai_services.prosecutor_planner import ProsecutorPlannerLLM
from backend.core.court_orchestrator import CourtOrchestrator
from backend.core.scoring_engine import compute_final_verdict
from backend.core.state_manager import StateManager
from backend.services.episode_loader import load_episode
from backend.schemas.trial import ProsecutorPlanMode

STAGE_ID = "stage_yamano_chain"


@pytest.fixture
def episode():
    return load_episode("turnabout_clock")


@pytest.mark.asyncio
async def test_start_court_starts_first_stage_without_legacy_round(episode):
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", "defense")
    events = await court.start_court(sid)
    types = [e["type"] for e in events]
    assert "stage_started" in types
    assert "witness_testimony" in types
    assert "round_started" not in types
    assert "prosecutor_response" not in types
    ts = await state.get_trial_state(sid)
    assert ts.current_stage_id == STAGE_ID
    assert ts.current_round_id is None


@pytest.mark.asyncio
async def test_witness_testimony_seeded(episode):
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", "defense")
    await court.start_court(sid)
    records = await state.get_court_records(sid)
    ids = [r.statement_id for r in records]
    assert "stmt_yamano_reported_at_two" in ids


@pytest.mark.asyncio
async def test_planner_only_picks_from_pool(episode):
    if not episode.prosecution_case:
        pytest.skip("turnabout_clock has no prosecution_case pool")
    planner = ProsecutorPlannerLLM(api_key=None)
    pool_ids = {c.claim_id for c in episode.prosecution_case.fixed_claim_pool}
    plan = await planner.plan(
        episode,
        available_claim_ids=["claim_presence"],
        used_claim_ids=[],
        weakened_claim_ids=[],
    )
    assert plan.selected_claim_id in pool_ids


@pytest.mark.asyncio
async def test_planner_skips_weakened(episode):
    if not episode.prosecution_case:
        pytest.skip("turnabout_clock has no prosecution_case pool")
    planner = ProsecutorPlannerLLM(api_key=None)
    plan = await planner.plan(
        episode,
        available_claim_ids=["claim_presence", "claim_cctv"],
        used_claim_ids=[],
        weakened_claim_ids=["claim_presence"],
    )
    assert plan.selected_claim_id != "claim_presence"


@pytest.mark.asyncio
async def test_mock_planner_uses_allowed_evidence_only(episode):
    if not episode.prosecution_case:
        pytest.skip("turnabout_clock has no prosecution_case pool")
    planner = ProsecutorPlannerLLM(api_key=None)
    plan = await planner.plan(
        episode,
        available_claim_ids=["claim_cctv", "claim_motive"],
        used_claim_ids=[],
        weakened_claim_ids=[],
    )
    allowed = set(episode.prosecution_case.allowed_evidence_ids)
    assert set(plan.selected_evidence_ids).issubset(allowed)


@pytest.mark.asyncio
async def test_good_stage_answer_damages_witness_and_gets_judge_comment(episode):
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", "defense")
    await state.add_evidence(sid, "autopsy_report")
    await court.start_court(sid)
    events = await court.process_defense_argument(
        sid,
        STAGE_ID,
        "부검 기록에 따르면 사망은 4시부터 5시인데 증인은 2시에 시체를 발견했다고 합니다.",
        ["autopsy_report"],
    )
    types = [e["type"] for e in events]
    assert "defense_argument_evaluated" in types
    assert "judge_comment" in types
    assert "witness_mental_update" in types
    ts = await state.get_trial_state(sid)
    assert ts.witness_mental_by_stage[STAGE_ID] == 75


@pytest.mark.asyncio
async def test_irrelevant_answer_increments_attempt(episode):
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", "defense")
    await court.start_court(sid)
    before = (await state.get_trial_state(sid)).stage_attempts.get(STAGE_ID, 0)
    events = await court.process_defense_argument(sid, STAGE_ID, "오늘 날씨 좋네요", [])
    after = (await state.get_trial_state(sid)).stage_attempts.get(STAGE_ID, 0)
    assert after == before + 1
    types = [e["type"] for e in events]
    assert "defense_argument_evaluated" in types
    assert "judge_comment" in types
    assert "prosecutor_pressure" in types


@pytest.mark.asyncio
async def test_request_hint(episode):
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", "defense")
    await court.start_court(sid)
    events = await court.request_hint(sid)
    assert events[0]["type"] == "helper_hint"
    assert events[0]["hint_level"] >= 1


def test_final_verdict_uses_score_ratio():
    assert compute_final_verdict(57, 60)["grade"] == "S"
    assert compute_final_verdict(45, 60)["grade"] == "A"
    assert compute_final_verdict(36, 60)["grade"] == "B"
    assert compute_final_verdict(35, 60)["grade"] == "F"


@pytest.mark.asyncio
async def test_full_playthrough_mock(episode):
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    sid = await state.create_session("turnabout_clock", "defense")
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
    for text, evs in answers:
        events = await court.process_defense_argument(sid, STAGE_ID, text, evs)
        if any(e.get("type") == "stage_cleared" for e in events):
            break

    ts = await state.get_trial_state(sid)
    meta = await state.get_meta(sid)
    assert meta.phase == "trial_finished"
    assert STAGE_ID in ts.cleared_stages
    assert any(e.get("type") == "ending" for e in events)
