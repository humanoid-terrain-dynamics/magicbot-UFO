# Repository Guidelines

## Project Structure & Module Organization

This repository is a self-contained Z1 (23-DoF) deployment bundle. Runtime entry points live in `scripts/`: `sim2sim_mujoco.py` runs the ONNX policy in MuJoCo, and `export_z1_deploy_artifact.py` stages/export artifacts from the source UFO checkout. Robot configuration and simulation assets are under `robot/` (`z1_23dof.yaml`, MJCF, and meshes). Checkpoints, ONNX models, tracking latents, contracts, and release hashes are under `model/z1_policy/`. Human-facing procedures are in `docs/`; generated rollout metrics belong in `outputs/`.

## Build, Test, and Development Commands

Run commands from the repository root with the project environment installed:

```powershell
uv run python scripts/export_z1_deploy_artifact.py --device cpu
uv run python scripts/sim2sim_mujoco.py --headless --latent zs_0.pkl --max-steps 250
uv run python scripts/sim2sim_mujoco.py --latent zs_0.pkl
```

The first command rebuilds the staged bundle (use `--device cuda` when exporting with a CUDA-capable environment). The headless command is the deterministic smoke test and updates `outputs/sim2sim_metrics.json`; omit `--headless` for an interactive viewer. The runner verifies release hashes and the `actor_obs [*, 631] -> action [*, 23]` contract before stepping.

## Coding Style & Naming Conventions

Python uses four-space indentation, type hints, `snake_case` functions/variables, and `PascalCase` classes. Keep scripts importable and put CLI behavior behind `main()` plus an `if __name__ == "__main__"` guard. Use UTF-8 text and small, focused changes; preserve JSON/YAML key names and the explicit 23-joint ordering.

## Testing Guidelines

There is no separate test suite in this bundle. Treat the 250-step headless MuJoCo run as the required regression check. Review `min_root_z`, `final_root_z`, `max_base_tilt_rad`, and `max_abs_torque` in `outputs/sim2sim_metrics.json`; investigate falls, excessive tilt, or torque saturation before longer runs.

## Commit & Pull Request Guidelines

This checkout has no available Git history, so use clear imperative commit subjects (for example, `Validate Z1 artifact contract`) and keep each commit focused. Pull requests should describe changed scripts/assets, state the exact smoke-test command and result, call out regenerated manifests or model files, and include relevant metrics or viewer screenshots. Do not claim sim-to-real validation from local sim-to-sim results.

## Security & Configuration Tips

Treat checkpoint and ONNX files as immutable release inputs. Export writes hashes to `model/z1_policy/release_manifest.json`; rerun the exporter rather than hand-editing staged artifacts. Do not commit secrets, local absolute paths, or unrelated generated files.
