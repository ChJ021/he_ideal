from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hetune.benchmarking import SEAL_PROFILE_COLUMNS, benchmark_supported_ckks_candidates, write_seal_profile
from hetune.operators.registry import build_default_registry


def test_benchmark_supported_ckks_candidates_covers_registry_support_set():
    registry = build_default_registry()
    ckks_config = {
        "backend_id": "seal_cpu",
        "ckks_param_id": "seal_profiled",
        "coefficient_modulus_chain": [60, 40, 40, 40, 40, 60],
        "bootstrapping_supported": False,
    }

    rows = benchmark_supported_ckks_candidates(ckks_config=ckks_config, registry=registry)

    expected = {
        provider.candidate_id
        for provider in registry.all()
        if provider.spec.supports_ckks_backend
    }
    assert {row["candidate_id"] for row in rows} == expected


def test_write_seal_profile_writes_expected_schema_and_metadata(tmp_path: Path):
    output = tmp_path / "seal_profile.csv"
    metadata = tmp_path / "seal_profile.metadata.json"
    ckks_config = {
        "backend_id": "seal_cpu",
        "ckks_param_id": "seal_profiled_bootstrap",
        "poly_modulus_degree": 32768,
        "coefficient_modulus_chain": [60, 40, 40, 40, 40, 40, 40, 60],
        "default_scale": "2^40",
        "bootstrapping_supported": True,
        "seal_source_path": "external/SEAL",
        "seal_build_path": "external/SEAL/build",
        "seal_install_path": "external/SEAL/install",
        "seal_version": "4.1.1",
        "benchmark_repetitions": 7,
        "benchmark_warmups": 2,
    }

    profile_path, metadata_path = write_seal_profile(
        ckks_config=ckks_config,
        output_path=output,
        metadata_path=metadata,
    )

    frame = pd.read_csv(profile_path)
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert tuple(frame.columns) == SEAL_PROFILE_COLUMNS
    assert frame["backend_id"].eq("seal_cpu").all()
    assert frame["ckks_param_id"].eq("seal_profiled_bootstrap").all()
    assert metadata_payload["seal_version"] == "4.1.1"
    assert metadata_payload["benchmark_repetitions"] == 7
    assert metadata_payload["benchmark_warmups"] == 2
