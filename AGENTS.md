# Repository Guidelines

## Project Structure & Module Organization

UFO is a Python 3.10 project managed with `uv`. Core package code lives in `humanoidverse/`, with training entry points such as `humanoidverse/train.py`, inference modules, agents, environments, tools, and robot/motion utilities. Hydra and robot/data configuration files live under `configs/` and `humanoidverse/config/`. Unit and smoke-style tests are in `tests/`. Documentation is in `README.md`, `README_zh-CN.md`, and `docs/`. Helper scripts live in `scripts/`, while top-level `tools/` contains standalone export utilities. Static site and paper assets are in `assets/` and `index.html`.

## Build, Test, and Development Commands

- `uv sync`: create or update the project environment from `pyproject.toml` and `uv.lock`.
- `uv run ruff check .`: run linting and import-order checks.
- `uv run python -m compileall -q humanoidverse tests`: verify Python files compile.
- `uv run python tests/test_motion_data_adapter.py`: run a focused unittest module.
- `bash scripts/smoke_release.sh`: run release smoke checks, including selected tests, compile checks, a short training run, and `git diff --check`.
- `./run_train.sh --agent fb --data-manifest configs/data/example_mix.yaml --gpu-ids single --smoke --work-dir /tmp/ufo_smoke_g1`: run the documented G1 smoke training path.

## Coding Style & Naming Conventions

Use 4-space indentation and Python type hints for new public helpers where practical. Follow existing module naming: lowercase files, descriptive function names, and test files named `test_*.py`. Ruff is configured for a 140-character line length, import sorting, and ignores `E402` and `E731`; do not reformat unrelated code. Keep Hydra YAML keys consistent with nearby config files.

## Testing Guidelines

Tests currently use Python `unittest` style and are executable as scripts. Add focused tests under `tests/` for motion-data adapters, robot config behavior, CLI argument handling, and inference/export dimensions. Prefer small fixture data generated in temporary directories unless an existing lightweight sample under `humanoidverse/data/examples/` fits the case.

## Commit & Pull Request Guidelines

Recent history uses short imperative commits, sometimes with conventional prefixes such as `fix:` or `docs:`. Keep commit subjects concise and behavior-focused, for example `fix: align robot observation dimensions`. Pull requests should describe the change, list commands run, note any dataset or GPU requirements, and include screenshots or videos only for documentation/site changes.

## Security & Configuration Tips

Do not commit checkpoints, W&B output, downloaded motion datasets, logs, cache directories, or local credentials. Use environment variables such as `WANDB_API_KEY`, `CUDA_VISIBLE_DEVICES`, and `UFO_CACHE_DIR` for machine-specific settings.
