from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from hetune.benchmarking import load_ckks_config, write_seal_profile
from hetune.deployment import HEDeploymentRunner
from hetune.experiments.distillation import DistillationRunner
from hetune.experiments.he_analysis import HEAnalysisRunner
from hetune.experiments.runner import ExperimentRunner

app = typer.Typer(help="HETune-LLM experiment CLI")
console = Console()


@app.command("run-activate")
def run_activate(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML"),
) -> None:
    """Run GELU and LayerNorm replacement search."""
    runner = ExperimentRunner(
        config,
        operator_scope="activation_norm",
        command_name="run-activate",
    )
    runner.run()
    console.print(
        f"[green]Completed activation/norm experiment[/green] {runner.experiment_id} -> {runner.paths.run_dir()}"
    )


@app.command("run-softmax")
def run_softmax(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML"),
) -> None:
    """Run Softmax-only replacement search."""
    runner = ExperimentRunner(
        config,
        operator_scope="softmax_only",
        command_name="run-softmax",
    )
    runner.run()
    console.print(
        f"[green]Completed softmax experiment[/green] {runner.experiment_id} -> {runner.paths.run_dir()}"
    )


@app.command("run-all")
def run_all(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML"),
) -> None:
    """Run all nonlinear replacement search: GELU, LayerNorm, and Softmax."""
    runner = ExperimentRunner(
        config,
        operator_scope="all_nonlinear",
        command_name="run-all",
    )
    runner.run()
    console.print(
        f"[green]Completed all-nonlinear experiment[/green] {runner.experiment_id} -> {runner.paths.run_dir()}"
    )


@app.command("run-all-he")
def run_all_he(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML"),
) -> None:
    """Run deployable HE-aware search for all nonlinear operators and export HE analysis."""
    runner = ExperimentRunner(
        config,
        operator_scope="all_nonlinear",
        command_name="run-all-he",
    )
    runner.run()
    he_output = HEAnalysisRunner(config, command_name="run-all-he").run()
    console.print(
        f"[green]Completed HE-aware all-nonlinear experiment[/green] {runner.experiment_id} -> {he_output}"
    )


@app.command("run-he")
def run_he(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML"),
) -> None:
    """Analyze generated schedules with imported HE microbenchmark costs."""
    runner = HEAnalysisRunner(config, command_name="run-he")
    output = runner.run()
    console.print(
        f"[green]Completed HE analysis[/green] {runner.experiment_id} -> {output}"
    )


@app.command("bench-seal-profile")
def bench_seal_profile(
    ckks_config: Path = typer.Option(..., "--ckks-config", help="CKKS YAML"),
    output: Path = typer.Option(..., "--output", help="Benchmark CSV output"),
    metadata_output: Path | None = typer.Option(
        None,
        "--metadata-output",
        help="Optional metadata JSON output",
    ),
    repetitions: int = typer.Option(10, "--repetitions", min=1),
    warmups: int = typer.Option(3, "--warmups", min=0),
) -> None:
    """Generate a standardized SEAL profile CSV for all CKKS-capable candidates."""
    _, ckks = load_ckks_config(ckks_config)
    profile_path, metadata_path = write_seal_profile(
        ckks_config=ckks,
        output_path=output,
        metadata_path=metadata_output,
        repetitions=repetitions,
        warmups=warmups,
    )
    console.print(
        "[green]Wrote SEAL profile[/green] "
        f"{profile_path} with metadata {metadata_path}"
    )


@app.command("deploy-he")
def deploy_he(
    config: Path = typer.Option(..., "--config", "-c", help="Deployment or experiment YAML"),
    allow_unavailable_backend: bool = typer.Option(
        False,
        "--allow-unavailable-backend",
        help="Write infeasible rows instead of failing when OpenFHE is not installed.",
    ),
) -> None:
    """Run true HE deployment cases through the configured OpenFHE backend."""
    runner = HEDeploymentRunner(
        config,
        allow_unavailable_backend=allow_unavailable_backend,
    )
    output = runner.run()
    console.print(f"[green]Completed HE deployment[/green] {output}")


@app.command()
def profile(config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML")) -> None:
    """Run layer-wise sensitivity profiling."""
    output = ExperimentRunner(config).profile()
    console.print(f"[green]Wrote profile[/green] {output}")


@app.command()
def tune(config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML")) -> None:
    """Generate uniform and greedy schedules."""
    output = ExperimentRunner(config).tune()
    console.print(f"[green]Wrote schedule[/green] {output}")


@app.command()
def evaluate(config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML")) -> None:
    """Evaluate generated schedules."""
    output = ExperimentRunner(config).evaluate()
    console.print(f"[green]Wrote metrics[/green] {output}")


@app.command()
def distill(config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML")) -> None:
    """Run standalone schedule distillation for the final generated schedule."""
    output = DistillationRunner(config).run()
    console.print(f"[green]Wrote distillation overrides[/green] {output}")
