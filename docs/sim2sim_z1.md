# Z1 Ordinary Sim2Sim

This is the Z1 equivalent of `docs/sim2sim_upstream.md`. It uses the staged Z1 model artifact and does not use the upstream G1 runtime, which is G1-29DoF-only.

## Build Artifact

Run from `UFO/` after the project environment is installed:

```powershell
uv run python ..\UFO_deploy\scripts\export_z1_deploy_artifact.py --device cpu
```

This stages the selected checkpoint, all `zs_*.pkl` tracking latents, the training-compatible 23-actuator Z1 MJCF/meshes, policy metadata, control contract, actor ONNX, backward-encoder ONNX, and SHA-256 manifest under `UFO_deploy/`.

## Terminal A: MuJoCo + Policy

```powershell
uv run python ..\UFO_deploy\scripts\sim2sim_mujoco.py --latent zs_0.pkl
```

Select the staged policy explicitly with `--policy-name`:

```powershell
uv run python ..\UFO_deploy\scripts\sim2sim_mujoco.py --policy-name z1_policy --latent zs_0.pkl
uv run python ..\UFO_deploy\scripts\sim2sim_mujoco.py --policy-name z1_policy_recovery5 --latent zs_0.pkl
```

The runner reads [`z1_clean20_md5_remote.txt`](z1_clean20_md5_remote.txt), copied from the source checkout, to display the semantic task name in both PowerShell and the MuJoCo window. For example, `zs_0.pkl` renders as `aini`; the source list maps `zs_0.pkl` through `zs_19.pkl` in order.

To override the manifest task name in the MuJoCo window and PowerShell, provide it explicitly:

```powershell
uv run python ..\UFO_deploy\scripts\sim2sim_mujoco.py --latent zs_0.pkl --task-name z1_20
```

For a noninteractive deterministic smoke rollout:

```powershell
uv run python ..\UFO_deploy\scripts\sim2sim_mujoco.py --policy-name z1_policy --headless --latent zs_0.pkl --max-steps 250
```

## Expected Result

The interactive command opens MuJoCo and executes the selected policy and tracking latent. It renders the manifest task name above the robot by default; `--task-name` overrides that label. The headless command exits with code zero and writes `UFO_deploy/outputs/sim2sim_<policy-name>_metrics.json`.

Interactive keyboard controls:

```text
i  interpolate to the default standing pose
]  enable the policy using the current tracking latent
[  start/restart tracking from frame zero
p  reset tracking to frame zero
o  stop policy action and hold the current joint targets
n  switch to the next staged tracking latent (`zs_0.pkl` ... `zs_19.pkl`)
```

Each policy has 20 selectable latent sequences; `n` switches the latent sequence, not the neural-network weights. The callback is available only in interactive mode (without `--headless`).

Before stepping, it verifies the deployment hashes, `actor_obs [batch, 631] -> action [batch, 23]` ONNX interface, 256-D finite latent, named 23-joint/23-actuator Z1 MJCF contract, and the deployed PD convention. The metrics contain `min_root_z`, `final_root_z`, `max_base_tilt_rad`, and `max_abs_torque`; use these to identify falls, excessive tilt, or torque saturation before attempting a longer rollout.

This validates local Z1 sim2sim only. It is not authorization or evidence for sim2real.
