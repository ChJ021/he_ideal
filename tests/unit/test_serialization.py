from hetune.core.ids import OperatorKey
from hetune.core.types import ScheduleEntry, SchedulePlan


def test_schedule_roundtrip_dict():
    operator = OperatorKey("m", 1, "gelu", "ffn", "a.b")
    schedule = SchedulePlan(
        metadata={"policy": "test"},
        entries=[ScheduleEntry(operator, "gelu.poly.degree3.v1")],
        constraints={"input_independent": True},
    )
    restored = SchedulePlan.from_dict(schedule.to_dict())
    assert restored.metadata["policy"] == "test"
    assert restored.entries[0].operator_key.id == operator.id
    assert restored.entries[0].candidate_id == "gelu.poly.degree3.v1"
