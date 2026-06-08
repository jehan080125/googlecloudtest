from backend.schemas.episode import TrialRound, TrialStage

HELPER_SUCCESS_LINES = [
    "우와! 성공이에요~~",
    "이 기세로 몰아붙여보자구요!",
]

DEFAULT_CONTRADICTION_BREAK_OPENERS = [
    "모순을 파헤쳤어요! 증인의 말이 바뀌었어요.",
    "또 말을 바꿨네요! 이번에도 틈을 찾아봐요.",
    "거의 다 왔어요! 마지막 모순까지 파고들어 보세요.",
]


def stage_hint_key(stage_id: str, phase: str) -> str:
    return f"{stage_id}:{phase}"


class Helper:
    @staticmethod
    def get_success_cheer_lines() -> list[str]:
        return list(HELPER_SUCCESS_LINES)

    def get_hint(self, current_round: TrialRound, hint_level: int) -> str | None:
        hints = current_round.hints
        if not hints:
            return "증거 상세를 다시 읽고, 검사 주장과 증인 증언의 연결 고리를 찾아보세요."
        if hint_level >= len(hints):
            return hints[-1]
        return hints[hint_level]

    @staticmethod
    def build_statement_chain(stage: TrialStage) -> list[str]:
        chain: list[str] = []
        for node in stage.fixed_testimony_chain:
            chain.append(node.statement_id)
            counter_id = node.counter_statement_id
            visited: set[str] = set()
            while counter_id and counter_id not in visited:
                visited.add(counter_id)
                chain.append(counter_id)
                counter = stage.counter_by_id(counter_id)
                counter_id = counter.next_counter_statement_id if counter else None
        return chain

    def get_statement_chain_index(self, stage: TrialStage, statement_id: str) -> int | None:
        chain = self.build_statement_chain(stage)
        try:
            return chain.index(statement_id)
        except ValueError:
            return None

    def get_current_contradiction_index(
        self,
        stage: TrialStage,
        *,
        current_counter_statement_id: str | None = None,
        current_testimony_id: str | None = None,
        usable_statement_ids: list[str] | None = None,
        stage_phase: str | None = None,
    ) -> int:
        if (
            stage.stage_id == "stage_epitaph_club"
            and stage_phase == "cross_exam_free"
            and "stmt_epitaph_isoeun_2" not in (usable_statement_ids or [])
        ):
            return 1
        statement_id = current_counter_statement_id or current_testimony_id
        if not statement_id and stage.fixed_testimony_chain:
            statement_id = stage.fixed_testimony_chain[0].statement_id
        if not statement_id:
            return 0
        index = self.get_statement_chain_index(stage, statement_id)
        return index if index is not None else 0

    def get_contradiction_hints(self, stage: TrialStage) -> list[str]:
        if stage.hints:
            return stage.hints
        phase_hints = stage.hints_by_phase.get("testimony")
        if phase_hints:
            return phase_hints
        return []

    def get_next_contradiction_hint(self, stage: TrialStage, broken_index: int) -> str | None:
        hints = self.get_contradiction_hints(stage)
        next_index = broken_index + 1
        if next_index >= len(hints):
            return None
        return hints[next_index]

    def get_helper_lines_after_break(self, stage: TrialStage, broken_index: int) -> list[str] | None:
        if stage.contradiction_helper_lines and broken_index < len(stage.contradiction_helper_lines):
            lines = stage.contradiction_helper_lines[broken_index]
            if lines:
                return lines

        next_hint = self.get_next_contradiction_hint(stage, broken_index)
        if not next_hint:
            return None

        opener = DEFAULT_CONTRADICTION_BREAK_OPENERS[
            min(broken_index, len(DEFAULT_CONTRADICTION_BREAK_OPENERS) - 1)
        ]
        return [opener, next_hint]

    def get_model_answer(self, stage: TrialStage, contradiction_index: int) -> str | None:
        answers = (stage.prosecution_context or {}).get("model_answers") or []
        if 0 <= contradiction_index < len(answers):
            return answers[contradiction_index]
        return None

    def get_model_answer_hint(self, stage: TrialStage, contradiction_index: int) -> str | None:
        hints = (stage.prosecution_context or {}).get("model_answer_hints") or []
        if 0 <= contradiction_index < len(hints):
            return hints[contradiction_index]
        answer = self.get_model_answer(stage, contradiction_index)
        if answer:
            return f"모범답안) {answer}"
        return None

    def get_stage_hint(
        self,
        stage: TrialStage,
        phase: str,
        hint_level: int,
        contradiction_index: int = 0,
    ) -> str:
        """Return the hint for the current contradiction only (not the next one)."""
        hints = self.get_contradiction_hints(stage)
        if hints:
            if 0 <= contradiction_index < len(hints):
                return hints[contradiction_index]
            return hints[-1]
        phase_hints = stage.hints_by_phase.get(phase)
        if phase_hints:
            phase_index = contradiction_index
            if phase == "cross_exam_free" and contradiction_index > 0:
                phase_index = contradiction_index - 1
            if 0 <= phase_index < len(phase_hints):
                return phase_hints[phase_index]
            return phase_hints[-1]
        return "현재 발언과 증거의 연결을 다시 보십시오."

    def get_phase_helper_lines(self, stage: TrialStage, phase: str) -> list[str] | None:
        """Legacy phase-based helper lines (manual hint context only)."""
        lines = stage.phase_helper_lines.get(phase)
        if lines:
            return lines
        return None
