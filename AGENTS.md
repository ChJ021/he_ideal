# Repository Guidelines

## Project Structure & Module Organization

HETune-LLM is a Python 3.12 package under `src/hetune/`. Core schedule, ID, cost, and serialization types live in `core/`; approximation providers are in `operators/`; HuggingFace discovery and replacement logic is in `models/`. Experiment orchestration, reports, diagnostics, and HE analysis are under `experiments/`, with profiling, scheduling, execution, cost, and security checks split into their matching packages.

YAML configuration is under `configs/`: experiments, models, datasets, approximation candidates, schedules, and CKKS profiles. Tests live in `tests/unit/` and `tests/integration/`. Documentation is in `docs/`. Generated artifacts belong in `outputs/runs/<experiment_id>/` or `outputs/cost_tables/`; keep raw/local data in `data/` and benchmarks in `benchmarks/`.

## Build, Test, and Development Commands

- `uv sync --extra dev`: install runtime and pytest dependencies.
- `uv run pytest`: run the full test suite.
- `uv run pytest tests/unit`: run fast unit tests.
- `uv run hetune run-activate --config configs/experiments/distilbert_sst2.yaml`: run GELU/LayerNorm scheduling.
- `uv run hetune run-softmax --config configs/experiments/distilbert_sst2_softmax.yaml`: run Softmax-only scheduling.
- `uv run hetune run-all --config configs/experiments/distilbert_sst2_all_nonlinear.yaml`: run all nonlinear candidates.
- `uv run hetune run-he --config configs/experiments/distilbert_sst2_all_nonlinear_he.yaml`: analyze schedules with imported HE costs.
- `uv build`: build distribution artifacts.

## Coding Style & Naming Conventions

Use 4-space indentation, type hints for public functions, `from __future__ import annotations`, and `dataclass(..., slots=True)` for structured records. Prefer `pathlib.Path` for paths. Keep imports grouped as standard library, third-party, then `hetune`. Use descriptive `snake_case` for Python names, YAML keys, and test names. No formatter is configured; match nearby style.

## Testing Guidelines

Pytest is configured in `pyproject.toml` with `tests` as the test root and `src` on `pythonpath`. Name test files `test_*.py` and functions `test_<behavior>()`. Add focused tests for registry, cost import, serialization, scheduling, HE analysis, path scoping, and operator candidates. Keep integration tests offline where possible.

## Commit & Pull Request Guidelines

This repository has no committed history yet. Use concise imperative commit messages with optional scope, such as `Add HE cost breakdown` or `Tighten softmax scheduler tests`. Pull requests should describe the experiment or code change, list touched configs and generated artifacts, include commands run, and summarize accuracy/cost tradeoffs.

## Security & Configuration Tips

Do not commit credentials, private checkpoints, or large raw datasets. Treat `outputs/` as generated evidence: update manifests, config snapshots, reports, and metrics intentionally. Preserve input-independent schedule validation when changing scheduler behavior.
