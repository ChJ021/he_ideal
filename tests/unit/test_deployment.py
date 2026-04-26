from pathlib import Path

import pandas as pd
import pytest
import torch

from hetune.core.ids import OperatorKey
from hetune.core.serialization import save_yaml
from hetune.core.types import CostVector, ExperimentPaths, ScheduleEntry, SchedulePlan
from hetune.deployment import (
    BackendAvailability,
    HEBackendUnavailableError,
    HEDeploymentRunner,
    OpenFHEExternalBackend,
    load_deployment_config,
)
from hetune.deployment.backend import DeploymentCaseRequest, DeploymentCaseResult
from hetune.deployment.forward_artifact import ForwardArtifact
from hetune.experiments.distillation import save_override_payload


class FakeDeploymentBackend:
    def availability(self) -> BackendAvailability:
        return BackendAvailability(True, "available", {})

    def run_case(self, request: DeploymentCaseRequest) -> DeploymentCaseResult:
        predictions = request.output_dir / "predictions.csv"
        pd.DataFrame(
            [
                {
                    "sample_id": 0,
                    "label": 1,
                    "prediction": 1,
                    "logit_0": -1.0,
                    "logit_1": 1.0,
                }
            ]
        ).to_csv(predictions, index=False)
        return DeploymentCaseResult(
            case_name=request.case_name,
            schedule_name=request.schedule_name,
            feasible=True,
            accuracy=1.0,
            sample_count=1,
            latency_ms=12.0,
            latency_p50_ms=11.0,
            latency_p95_ms=13.0,
            cost=CostVector(
                latency_ms=12.0,
                rotations=3,
                ct_ct_mults=4,
                ct_pt_mults=5,
                rescale_count=6,
                relin_count=7,
                depth=8,
                bootstrap_count=1,
                memory_mb=9.0,
            ),
            predictions_path=predictions,
            backend_metadata={"backend": "fake"},
        )


class MissingBackend:
    def availability(self) -> BackendAvailability:
        return BackendAvailability(False, "missing backend", {"install_dir": "/missing"})


class CapturingForwardBackend:
    def __init__(self) -> None:
        self.requests: list[DeploymentCaseRequest] = []

    def availability(self) -> BackendAvailability:
        return BackendAvailability(True, "available", {})

    def run_case(self, request: DeploymentCaseRequest) -> DeploymentCaseResult:
        self.requests.append(request)
        predictions = request.output_dir / "predictions.csv"
        pd.DataFrame(
            [{"sample_id": 0, "label": 1, "prediction": 1, "logit_0": -0.5, "logit_1": 0.5}]
        ).to_csv(predictions, index=False)
        return DeploymentCaseResult(
            case_name=request.case_name,
            schedule_name=request.schedule_name,
            feasible=True,
            accuracy=1.0,
            sample_count=request.sample_size,
            predictions_path=predictions,
            backend_metadata={
                "runner_mode": "openfhe_distilbert_forward",
                "accuracy_source": "native_decrypted_logits",
            },
        )


class NoNativeAccuracyForwardBackend:
    def availability(self) -> BackendAvailability:
        return BackendAvailability(True, "available", {})

    def run_case(self, request: DeploymentCaseRequest) -> DeploymentCaseResult:
        return DeploymentCaseResult(
            case_name=request.case_name,
            schedule_name=request.schedule_name,
            feasible=True,
            backend_metadata={
                "runner_mode": "openfhe_distilbert_forward",
                "accuracy_source": "plaintext_evaluation_metrics_if_available",
            },
        )


def test_load_deployment_config_resolves_experiment_and_backend(tmp_path: Path):
    deployment_path = _write_deployment_fixture(tmp_path)

    loaded = load_deployment_config(deployment_path)

    assert loaded.deployment_id == "mock_deploy"
    assert loaded.experiment.experiment["experiment_id"] == "mock_exp"
    assert loaded.backend_config["install_dir"] == str(tmp_path / "openfhe" / "install")
    assert loaded.cases == ["high", "pre_distill", "post_distill"]


def test_deployment_runner_writes_three_case_comparison(tmp_path: Path):
    deployment_path = _write_deployment_fixture(tmp_path)

    comparison_path = HEDeploymentRunner(
        deployment_path,
        backend=FakeDeploymentBackend(),
    ).run()

    frame = pd.read_csv(comparison_path)
    assert list(frame["case"]) == ["high", "pre_distill", "post_distill"]
    assert frame["feasible"].all()
    assert frame["accuracy"].tolist() == [1.0, 1.0, 1.0]
    assert (comparison_path.parent / "deployment_report.md").exists()
    assert (comparison_path.parent / "post_distill" / "metrics.csv").exists()


def test_deployment_runner_fails_fast_when_backend_missing(tmp_path: Path):
    deployment_path = _write_deployment_fixture(tmp_path)

    with pytest.raises(HEBackendUnavailableError, match="missing backend"):
        HEDeploymentRunner(deployment_path, backend=MissingBackend()).run()


def test_deployment_runner_can_write_infeasible_rows_for_missing_backend(tmp_path: Path):
    deployment_path = _write_deployment_fixture(tmp_path, fail_on_unavailable_backend=False)

    comparison_path = HEDeploymentRunner(
        deployment_path,
        backend=MissingBackend(),
    ).run()
    frame = pd.read_csv(comparison_path)

    assert not frame["feasible"].any()
    assert set(frame["error"]) == {"missing backend"}


def test_post_distill_case_requires_overrides(tmp_path: Path):
    deployment_path = _write_deployment_fixture(tmp_path, write_overrides=False)

    with pytest.raises(FileNotFoundError, match="requires overrides"):
        HEDeploymentRunner(deployment_path, backend=FakeDeploymentBackend()).run()


def test_openfhe_backend_uses_single_root_layout(tmp_path: Path):
    root = tmp_path / "openfhe"
    backend = OpenFHEExternalBackend({"openfhe_root": str(root)})

    availability = backend.availability()

    assert not availability.available
    assert availability.checked_paths["source_dir"] == str(root / "src")
    assert availability.checked_paths["build_dir"] == str(root / "build")
    assert availability.checked_paths["install_dir"] == str(root / "install")
    assert availability.checked_paths["python_dir"] == str(root / "openfhe-python")


def test_openfhe_backend_resolves_relative_project_runner_path(tmp_path: Path):
    config_dir = tmp_path / "configs" / "he_backend"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "openfhe.yaml"
    backend = OpenFHEExternalBackend(
        {
            "openfhe_root": str(tmp_path / "openfhe"),
            "runner_path": "../../build/openfhe_runner/hetune_openfhe_runner",
        },
        config_path=config_path,
    )

    assert backend.runner_path == tmp_path / "build" / "openfhe_runner" / "hetune_openfhe_runner"


def test_forward_mode_passes_manifest_and_requires_native_accuracy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    deployment_path = _write_deployment_fixture(
        tmp_path,
        deployment_updates={
            "runner_mode": "openfhe_distilbert_forward",
            "encrypted_sample_size": 2,
            "encrypted_sequence_length": 4,
            "fail_on_plaintext_accuracy_fallback": True,
            "linear_kernel": "bsgs_hoisted",
            "bsgs_baby_step": 16,
            "fuse_qkv": True,
            "packing_strategy": "row_packed",
            "profile_native_stages": True,
        },
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    def fake_export(**kwargs):
        assert kwargs["sample_size"] == 2
        assert kwargs["sequence_length"] == 4
        return ForwardArtifact(manifest_path=manifest_path, sample_count=2, sequence_length=4, hidden_size=8)

    monkeypatch.setattr("hetune.deployment.runner.export_distilbert_forward_artifact", fake_export)
    backend = CapturingForwardBackend()

    comparison_path = HEDeploymentRunner(deployment_path, backend=backend).run()

    assert len(backend.requests) == 3
    assert all(request.runner_mode == "openfhe_distilbert_forward" for request in backend.requests)
    assert all(request.forward_manifest_path == manifest_path for request in backend.requests)
    assert all(request.sample_size == 2 for request in backend.requests)
    assert all(request.sequence_length == 4 for request in backend.requests)
    assert all(request.multiplicative_depth == 33 for request in backend.requests)
    assert all(request.scaling_mod_size == 44 for request in backend.requests)
    assert all(request.first_mod_size == 55 for request in backend.requests)
    assert all(request.poly_modulus_degree == 4096 for request in backend.requests)
    assert all(request.to_dict()["multiplicative_depth"] == 33 for request in backend.requests)
    assert all(request.linear_kernel == "bsgs_hoisted" for request in backend.requests)
    assert all(request.bsgs_baby_step == 16 for request in backend.requests)
    assert all(request.fuse_qkv for request in backend.requests)
    assert all(request.packing_strategy == "row_packed" for request in backend.requests)
    assert all(request.to_dict()["profile_native_stages"] for request in backend.requests)
    frame = pd.read_csv(comparison_path)
    assert set(frame["accuracy_source"]) == {"native_decrypted_logits"}


def test_forward_mode_rejects_plaintext_accuracy_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    deployment_path = _write_deployment_fixture(
        tmp_path,
        deployment_updates={
            "runner_mode": "openfhe_distilbert_forward",
            "fail_on_plaintext_accuracy_fallback": True,
        },
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "hetune.deployment.runner.export_distilbert_forward_artifact",
        lambda **_: ForwardArtifact(manifest_path=manifest_path, sample_count=1, sequence_length=8, hidden_size=8),
    )

    comparison_path = HEDeploymentRunner(
        deployment_path,
        backend=NoNativeAccuracyForwardBackend(),
    ).run()

    frame = pd.read_csv(comparison_path)
    assert not frame["feasible"].any()
    assert frame["error"].str.contains("native decrypted-logits accuracy").all()


def _write_deployment_fixture(
    tmp_path: Path,
    *,
    fail_on_unavailable_backend: bool = True,
    write_overrides: bool = True,
    deployment_updates: dict[str, object] | None = None,
) -> Path:
    output_root = tmp_path / "outputs"
    save_yaml({"model_id": "mock", "model_name_or_path": "unused"}, tmp_path / "model.yaml")
    save_yaml(
        {
            "dataset_id": "mock_data",
            "dataset_name": "unused",
            "calibration_split": "train",
            "validation_split": "validation",
            "text_fields": ["sentence"],
        },
        tmp_path / "dataset.yaml",
    )
    save_yaml(
        {"candidates": [{"candidate_id": "gelu.poly.degree2.v1", "enabled": True}]},
        tmp_path / "approximations.yaml",
    )
    save_yaml(
        {
            "ckks_param_id": "mock_ckks",
            "backend": "static-only",
            "backend_id": "mock_backend",
            "security_bits": 128,
        },
        tmp_path / "ckks.yaml",
    )
    save_yaml(
        {
            "experiment_id": "mock_exp",
            "operator_scope": "activation_norm",
            "operator_types": ["gelu"],
            "model_config": "model.yaml",
            "dataset_config": "dataset.yaml",
            "approximation_config": "approximations.yaml",
            "ckks_config": "ckks.yaml",
            "output_root": str(output_root),
            "sequence_length": 8,
        },
        tmp_path / "experiment.yaml",
    )
    save_yaml(
        {
            "backend": "openfhe_external",
            "backend_id": "mock_openfhe",
            "openfhe_root": str(tmp_path / "openfhe"),
            "install_dir": str(tmp_path / "openfhe" / "install"),
            "ckks": {
                "multiplicative_depth": 33,
                "scaling_mod_size": 44,
                "first_mod_size": 55,
                "poly_modulus_degree": 4096,
            },
        },
        tmp_path / "openfhe.yaml",
    )
    deployment_raw = {
        "deployment_id": "mock_deploy",
        "experiment_config": "experiment.yaml",
        "he_backend_config": "openfhe.yaml",
        "cases": ["high", "pre_distill", "post_distill"],
        "sample_size": 1,
        "sequence_length": 8,
        "latency_repetitions": 1,
        "fail_on_unavailable_backend": fail_on_unavailable_backend,
    }
    deployment_raw.update(deployment_updates or {})
    save_yaml(deployment_raw, tmp_path / "deployment.yaml")

    paths = ExperimentPaths("mock_exp", root=output_root)
    paths.ensure()
    operator = OperatorKey("mock", 0, "gelu", "ffn_activation", "layer.0.act")
    high = SchedulePlan(
        metadata={"policy": "uniform_high"},
        entries=[ScheduleEntry(operator, "gelu.exact.high.v1")],
    )
    generated = SchedulePlan(
        metadata={"policy": "validated_greedy"},
        entries=[ScheduleEntry(operator, "gelu.poly.degree2.v1")],
    )
    save_yaml(high.to_dict(), paths.schedule_dir() / "uniform_high.yaml")
    save_yaml(generated.to_dict(), paths.schedule_dir() / "hetune_generated.yaml")
    if write_overrides:
        save_override_payload(
            {
                "schedule_name": "hetune_generated",
                "entries": [
                    {
                        "operator_id": operator.id,
                        "operator_path": operator.path,
                        "candidate_id": "layernorm.exact.high.v1",
                        "parameter_name": "weight",
                        "tensor": torch.ones(1),
                    }
                ],
            },
            paths.distillation_dir() / "overrides.pt",
        )
    return tmp_path / "deployment.yaml"
