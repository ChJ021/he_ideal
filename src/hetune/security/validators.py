from __future__ import annotations

from hetune.core.types import SchedulePlan


class SecurityValidator:
    """MVP audit checks for fixed, input-independent schedules."""

    def validate(self, schedule: SchedulePlan) -> list[str]:
        findings: list[str] = []
        if not schedule.entries:
            findings.append("schedule has no entries")
        if schedule.constraints.get("input_independent") is False:
            findings.append("schedule is marked input-dependent")
        for entry in schedule.entries:
            if not entry.candidate_id:
                findings.append(f"{entry.operator_key.id} has no candidate_id")
        return findings
