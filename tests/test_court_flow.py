import pytest

from backend.core.court_orchestrator import CourtOrchestrator
from backend.core.state_manager import StateManager

STAGE_ID = "stage_yamano_chain"


@pytest.mark.asyncio
async def test_stage_flow_with_evidence():
    state = StateManager()
    court = CourtOrchestrator(state, api_key=None)
    session_id = await state.create_session("turnabout_clock", "defense")
    await state.add_evidence(session_id, "autopsy_report")
    events = await court.start_court(session_id)
    assert any(e.get("type") == "stage_started" for e in events)
    assert not any(e.get("type") == "round_started" for e in events)

    result = await court.process_defense_argument(
        session_id,
        STAGE_ID,
        "부검 기록상 사망은 4시부터 5시인데 증인은 2시에 시체를 발견했다고 합니다.",
        ["autopsy_report"],
    )
    assert any(e.get("type") == "defense_argument_evaluated" for e in result)
    assert any(e.get("type") == "judge_comment" for e in result)
