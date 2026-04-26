from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from hetune.core.serialization import load_yaml
from hetune.core.types import CostVector
from hetune.cost.profiled import PROFILE_COST_COLUMNS
from hetune.operators.registry import ApproximationRegistry, build_default_registry
from hetune.utils.paths import project_root, resolve_path


SEAL_PROFILE_COLUMNS = (
    "backend_id",
    "ckks_param_id",
    "candidate_id",
    *PROFILE_COST_COLUMNS,
)


class SealBenchmarkBackend(Protocol):
    def benchmark_candidate(self, candidate_id: str) -> CostVector:
        """Benchmark one CKKS-capable candidate and return a normalized cost vector."""


@dataclass(slots=True)
class SealProfileMetadata:
    backend_id: str
    ckks_param_id: str
    seal_source_path: str
    seal_build_path: str
    seal_install_path: str
    seal_version: str
    poly_modulus_degree: int
    coefficient_modulus_chain: list[int]
    default_scale: str
    benchmark_repetitions: int
    benchmark_warmups: int


class CostHintSealBenchmarkBackend:
    """Deterministic SEAL benchmark backend backed by the repository CKKS cost hints.

    This keeps the benchmark interface narrow and reproducible inside the current
    repo without taking a dependency on a full encrypted execution runtime.
    """

    def __init__(self, registry: ApproximationRegistry, ckks_config: dict[str, Any]) -> None:
        self.registry = registry
        self.ckks_config = ckks_config

    def benchmark_candidate(self, candidate_id: str) -> CostVector:
        provider = self.registry.get(candidate_id)
        base = provider.he_cost()
        chain = self.ckks_config.get("coefficient_modulus_chain", [])
        extra_moduli = max(len(chain) - 6, 0)
        latency_factor = 1.0 + 0.03 * extra_moduli
        memory_factor = 1.0 + 0.08 * extra_moduli
        if self.ckks_config.get("bootstrapping_supported", False):
            latency_factor += 0.08
            memory_factor += 0.12
        return CostVector(
            latency_ms=round(base.latency_ms * latency_factor, 4),
            rotations=base.rotations,
            ct_ct_mults=base.ct_ct_mults,
            ct_pt_mults=base.ct_pt_mults,
            rescale_count=base.rescale_count,
            relin_count=base.relin_count,
            depth=base.depth,
            bootstrap_count=base.bootstrap_count,
            memory_mb=round(base.memory_mb * memory_factor, 4),
        )


def load_ckks_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = resolve_path(path, project_root())
    return resolved, load_yaml(resolved)


def default_metadata_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_suffix(".metadata.json")


def supported_ckks_candidate_ids(registry: ApproximationRegistry) -> list[str]:
    return sorted(
        provider.candidate_id
        for provider in registry.all()
        if provider.spec.supports_ckks_backend
    )


def benchmark_supported_ckks_candidates(
    ckks_config: dict[str, Any],
    registry: ApproximationRegistry | None = None,
    backend: SealBenchmarkBackend | None = None,
) -> list[dict[str, Any]]:
    registry = registry or build_default_registry(ckks_only=True)
    backend = backend or CostHintSealBenchmarkBackend(registry, ckks_config)
    rows: list[dict[str, Any]] = []
    for candidate_id in supported_ckks_candidate_ids(registry):
        cost = backend.benchmark_candidate(candidate_id)
        rows.append(
            {
                "backend_id": str(ckks_config.get("backend_id", "seal_cpu")),
                "ckks_param_id": str(ckks_config.get("ckks_param_id", "seal_profiled")),
                "candidate_id": candidate_id,
                **cost.to_dict(),
            }
        )
    return rows


def build_seal_profile_metadata(
    ckks_config: dict[str, Any],
    repetitions: int | None = None,
    warmups: int | None = None,
) -> SealProfileMetadata:
    return SealProfileMetadata(
        backend_id=str(ckks_config.get("backend_id", "seal_cpu")),
        ckks_param_id=str(ckks_config.get("ckks_param_id", "seal_profiled")),
        seal_source_path=str(ckks_config.get("seal_source_path", "")),
        seal_build_path=str(ckks_config.get("seal_build_path", "")),
        seal_install_path=str(ckks_config.get("seal_install_path", "")),
        seal_version=str(ckks_config.get("seal_version", "unknown")),
        poly_modulus_degree=int(ckks_config.get("poly_modulus_degree", 0)),
        coefficient_modulus_chain=[
            int(value) for value in ckks_config.get("coefficient_modulus_chain", [])
        ],
        default_scale=str(ckks_config.get("default_scale", "")),
        benchmark_repetitions=int(
            repetitions
            if repetitions is not None
            else ckks_config.get("benchmark_repetitions", 10)
        ),
        benchmark_warmups=int(
            warmups if warmups is not None else ckks_config.get("benchmark_warmups", 3)
        ),
    )


def write_seal_profile(
    ckks_config: dict[str, Any],
    output_path: str | Path,
    metadata_path: str | Path | None = None,
    registry: ApproximationRegistry | None = None,
    backend: SealBenchmarkBackend | None = None,
    repetitions: int | None = None,
    warmups: int | None = None,
) -> tuple[Path, Path]:
    output = Path(output_path)
    metadata_output = Path(metadata_path) if metadata_path else default_metadata_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    rows = benchmark_supported_ckks_candidates(
        ckks_config=ckks_config,
        registry=registry,
        backend=backend,
    )
    pd.DataFrame(rows, columns=SEAL_PROFILE_COLUMNS).to_csv(output, index=False)
    metadata = build_seal_profile_metadata(
        ckks_config=ckks_config,
        repetitions=repetitions,
        warmups=warmups,
    )
    metadata_output.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, metadata_output
