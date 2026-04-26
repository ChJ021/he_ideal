from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hetune.core.ids import OperatorKey


@dataclass(slots=True)
class CostVector:
    latency_ms: float = 0.0
    rotations: int = 0
    ct_ct_mults: int = 0
    ct_pt_mults: int = 0
    rescale_count: int = 0
    relin_count: int = 0
    depth: int = 0
    bootstrap_count: int = 0
    memory_mb: float = 0.0

    def weighted(self, weights: dict[str, float] | None = None) -> float:
        weights = weights or {}
        return (
            self.latency_ms * weights.get("latency_ms", 1.0)
            + self.rotations * weights.get("rotations", 0.05)
            + self.ct_ct_mults * weights.get("ct_ct_mults", 1.0)
            + self.ct_pt_mults * weights.get("ct_pt_mults", 0.15)
            + self.rescale_count * weights.get("rescale_count", 0.4)
            + self.relin_count * weights.get("relin_count", 0.4)
            + self.depth * weights.get("depth", 1.5)
            + self.bootstrap_count * weights.get("bootstrap_count", 50.0)
            + self.memory_mb * weights.get("memory_mb", 0.01)
        )

    def __add__(self, other: "CostVector") -> "CostVector":
        return CostVector(
            latency_ms=self.latency_ms + other.latency_ms,
            rotations=self.rotations + other.rotations,
            ct_ct_mults=self.ct_ct_mults + other.ct_ct_mults,
            ct_pt_mults=self.ct_pt_mults + other.ct_pt_mults,
            rescale_count=self.rescale_count + other.rescale_count,
            relin_count=self.relin_count + other.relin_count,
            depth=self.depth + other.depth,
            bootstrap_count=self.bootstrap_count + other.bootstrap_count,
            memory_mb=self.memory_mb + other.memory_mb,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "latency_ms": self.latency_ms,
            "rotations": self.rotations,
            "ct_ct_mults": self.ct_ct_mults,
            "ct_pt_mults": self.ct_pt_mults,
            "rescale_count": self.rescale_count,
            "relin_count": self.relin_count,
            "depth": self.depth,
            "bootstrap_count": self.bootstrap_count,
            "memory_mb": self.memory_mb,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object] | None) -> "CostVector":
        data = data or {}
        return cls(
            latency_ms=float(data.get("latency_ms", 0.0)),
            rotations=int(data.get("rotations", 0)),
            ct_ct_mults=int(data.get("ct_ct_mults", 0)),
            ct_pt_mults=int(data.get("ct_pt_mults", 0)),
            rescale_count=int(data.get("rescale_count", 0)),
            relin_count=int(data.get("relin_count", 0)),
            depth=int(data.get("depth", 0)),
            bootstrap_count=int(data.get("bootstrap_count", 0)),
            memory_mb=float(data.get("memory_mb", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class ApproximationSpec:
    operator_type: str
    candidate_id: str
    approximation_family: str
    quality_rank: int
    degree: int = 0
    valid_input_range: tuple[float, float] | None = None
    depth: int = 0
    rotation_requirement: int = 0
    scale_requirement: str = "2^40"
    supports_plaintext_simulation: bool = True
    supports_ckks_backend: bool = False
    expected_accuracy_risk: float = 0.0
    implementation_backend: str = "torch"
    cost_hint: CostVector = field(default_factory=CostVector)

    def to_dict(self) -> dict[str, object]:
        return {
            "operator_type": self.operator_type,
            "candidate_id": self.candidate_id,
            "approximation_family": self.approximation_family,
            "quality_rank": self.quality_rank,
            "degree": self.degree,
            "valid_input_range": list(self.valid_input_range)
            if self.valid_input_range
            else None,
            "depth": self.depth,
            "rotation_requirement": self.rotation_requirement,
            "scale_requirement": self.scale_requirement,
            "supports_plaintext_simulation": self.supports_plaintext_simulation,
            "supports_ckks_backend": self.supports_ckks_backend,
            "expected_accuracy_risk": self.expected_accuracy_risk,
            "implementation_backend": self.implementation_backend,
            "cost_hint": self.cost_hint.to_dict(),
        }


@dataclass(slots=True)
class ScheduleEntry:
    operator_key: OperatorKey
    candidate_id: str
    ckks_param_id: str = "static_ckks_128"
    scale_id: str = "scale_2_40"
    level_budget: int = 0
    bootstrap_policy: str = "none"
    layout_id: str = "plaintext_sim"

    def to_dict(self) -> dict[str, object]:
        return {
            "operator_key": self.operator_key.to_dict(),
            "operator_id": self.operator_key.id,
            "layer_index": self.operator_key.layer_index,
            "operator_type": self.operator_key.operator_type,
            "candidate_id": self.candidate_id,
            "ckks_param_id": self.ckks_param_id,
            "scale_id": self.scale_id,
            "level_budget": self.level_budget,
            "bootstrap_policy": self.bootstrap_policy,
            "layout_id": self.layout_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ScheduleEntry":
        return cls(
            operator_key=OperatorKey.from_dict(data["operator_key"]),  # type: ignore[arg-type]
            candidate_id=str(data["candidate_id"]),
            ckks_param_id=str(data.get("ckks_param_id", "static_ckks_128")),
            scale_id=str(data.get("scale_id", "scale_2_40")),
            level_budget=int(data.get("level_budget", 0)),
            bootstrap_policy=str(data.get("bootstrap_policy", "none")),
            layout_id=str(data.get("layout_id", "plaintext_sim")),
        )


@dataclass(slots=True)
class SchedulePlan:
    metadata: dict[str, Any]
    entries: list[ScheduleEntry]
    constraints: dict[str, Any] = field(default_factory=dict)

    def entry_for(self, operator_id: str) -> ScheduleEntry | None:
        return next(
            (entry for entry in self.entries if entry.operator_key.id == operator_id),
            None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metadata": self.metadata,
            "entries": [entry.to_dict() for entry in self.entries],
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SchedulePlan":
        return cls(
            metadata=dict(data.get("metadata", {})),
            entries=[
                ScheduleEntry.from_dict(item)
                for item in data.get("entries", [])  # type: ignore[union-attr]
            ],
            constraints=dict(data.get("constraints", {})),
        )


@dataclass(slots=True)
class SensitivityRecord:
    operator_key: OperatorKey
    candidate_id: str
    baseline_accuracy: float
    candidate_accuracy: float
    accuracy_drop: float
    logit_kl: float
    label_flip_rate: float
    hidden_l2: float = 0.0
    attention_kl: float = 0.0

    @property
    def sensitivity_score(self) -> float:
        return (
            max(self.accuracy_drop, 0.0)
            + self.logit_kl
            + self.label_flip_rate
            + self.hidden_l2
            + self.attention_kl
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "operator_id": self.operator_key.id,
            "layer_index": self.operator_key.layer_index,
            "operator_type": self.operator_key.operator_type,
            "operator_name": self.operator_key.name,
            "operator_path": self.operator_key.path,
            "candidate_id": self.candidate_id,
            "baseline_accuracy": self.baseline_accuracy,
            "candidate_accuracy": self.candidate_accuracy,
            "accuracy_drop": self.accuracy_drop,
            "logit_kl": self.logit_kl,
            "label_flip_rate": self.label_flip_rate,
            "hidden_l2": self.hidden_l2,
            "attention_kl": self.attention_kl,
            "sensitivity_score": self.sensitivity_score,
        }


@dataclass(slots=True)
class ExperimentPaths:
    experiment_id: str
    root: Path = Path("outputs")

    def run_dir(self) -> Path:
        return self.root / "runs" / self.experiment_id

    def config_dir(self) -> Path:
        return self.run_dir() / "configs"

    def profile_dir(self) -> Path:
        return self.run_dir() / "profiles"

    def schedule_dir(self) -> Path:
        return self.run_dir() / "schedules"

    def evaluation_dir(self) -> Path:
        return self.run_dir() / "evaluations"

    def he_analysis_dir(self) -> Path:
        return self.run_dir() / "he_analysis"

    def he_deployment_dir(self) -> Path:
        return self.run_dir() / "he_deployment"

    def distillation_dir(self) -> Path:
        return self.run_dir() / "distillation"

    def figure_dir(self) -> Path:
        return self.run_dir() / "figures"

    def report_dir(self) -> Path:
        return self.run_dir() / "reports"

    def log_dir(self) -> Path:
        return self.run_dir() / "logs"

    def manifest_path(self) -> Path:
        return self.run_dir() / "manifest.yaml"

    def ensure(self) -> None:
        for directory in (
            self.run_dir(),
            self.config_dir(),
            self.profile_dir(),
            self.schedule_dir(),
            self.evaluation_dir(),
            self.he_analysis_dir(),
            self.he_deployment_dir(),
            self.distillation_dir(),
            self.figure_dir(),
            self.report_dir(),
            self.log_dir(),
        ):
            directory.mkdir(parents=True, exist_ok=True)
