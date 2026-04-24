from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OperatorKey:
    """Stable identifier for a replaceable model operator."""

    model_id: str
    layer_index: int
    operator_type: str
    name: str
    path: str

    @property
    def id(self) -> str:
        return (
            f"{self.model_id}.layer{self.layer_index}."
            f"{self.operator_type}.{self.name}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "layer_index": self.layer_index,
            "operator_type": self.operator_type,
            "name": self.name,
            "path": self.path,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "OperatorKey":
        return cls(
            model_id=str(data["model_id"]),
            layer_index=int(data["layer_index"]),
            operator_type=str(data["operator_type"]),
            name=str(data["name"]),
            path=str(data["path"]),
        )
