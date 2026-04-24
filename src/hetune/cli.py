from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

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
