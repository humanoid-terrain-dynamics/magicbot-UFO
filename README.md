# Magicbot Z1 UFO: 23-DoF FB Motion Priors

This repository is a Magicbot Z1 workstream built on top of RoboParty Lab's
UFO framework. The current focus is Z1 23-DoF Forward-Backward (FB) training,
motion-prior evaluation, ONNX export, and a local MuJoCo slider for inspecting
44 learned motion priors.

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh-CN.md">Chinese</a>
</p>

<p align="center">
  <img src="./assets/z1_ufo_8x4_labeled.gif" alt="Magicbot Z1 UFO FB motion-tracking results across 32 labeled clips" width="760" />
</p>

## Outcome First

The visible result above is a Z1 FB policy tracking 32 labeled clips from the
clean Z1 motion set. The policy is trained as a latent-conditioned motion prior:
instead of one task reward per skill, UFO learns a single actor conditioned by
latent vectors `z`. At inference time, a reference motion is encoded into a
trajectory of `z` vectors, and the actor follows that motion in simulation.

The interactive local artifact is:

```powershell
.venv\Scripts\python.exe tools\z1\ufo_44_slider_tk_mujoco.py `
  --artifact-dir ..\checkpoints\Z1_UFO\compare_hand_vs_nohand\fakehand_32M
```

The slider lets you inspect motion indices `0..43`, nudge with Left/Right,
reset with `R`, pause with Space, and quit with `Q`.

## Prior to Outcome

The Z1 outcome is driven by a curated motion prior rather than a hand-authored
task controller.

| Stage | Local contract | Output |
| --- | --- | --- |
| Robot definition | `configs/robots/z1_23dof.yaml` and Z1 MJCF assets | 23-DoF joint order, PD gains, limits, action scale |
| Motion prior | `configs/data/z1_mimic_clean.yaml` | 44 curated RobotState clips |
| FB training | `run_train.sh --agent fb --robot-config configs/robots/z1_23dof.yaml` | rolling checkpoint and evaluation metrics |
| Tracking inference | `humanoidverse.tracking_inference` | MP4s, `zs_*.pkl`, ONNX policy, `.meta.json` |
| Local sim-to-sim | `tools/z1/deploy_onnx_mujoco.py` | plain MuJoCo rollout using the exported policy |
| Interactive prior browser | `tools/z1/ufo_44_slider_tk_mujoco.py` | Tkinter slider over 44 latent trajectories |

The 44-motion slider loads four cached motion groups:

```text
cache/motion_data/z1_mimic_clean/z1_cyclic_locomotion_train_near10s_ufo.pkl
cache/motion_data/z1_mimic_clean/z1_atomic_skills_train_near10s_ufo.pkl
cache/motion_data/z1_mimic_clean/z1_acrobatics_recovery_train_near10s_ufo.pkl
cache/motion_data/z1_mimic_clean/z1_pose_low_motion_train_near10s_ufo.pkl
```

Generate the matching latent cache before launching the slider:

```powershell
$env:MUJOCO_GL = "glfw"
$env:PYTHONUTF8 = "1"

.venv\Scripts\python.exe tools\z1\generate_z1_44_latents.py `
  --model-folder ..\checkpoints\Z1_UFO\z1_mimic_clean_fb_2xa100_1024env_20260716_044447 `
  --artifact-dir ..\checkpoints\Z1_UFO\compare_hand_vs_nohand\fakehand_32M `
  --device cpu
```

## Z1 Artifact Contract

Large checkpoints, ONNX files, videos, latent caches, and motion caches are not
stored in Git. Keep them under the repository parent checkpoint layout:

```text
../checkpoints/Z1_UFO/
```

The public Z1 dataset/checkpoint mirror is:

```text
https://huggingface.co/datasets/PhangHongHao/UFO-Z1
```

Download the best Z1 FB checkpoint into the local checkpoint layout:

```bash
hf download PhangHongHao/UFO-Z1 \
  --repo-type dataset \
  --local-dir ../checkpoints/Z1_UFO \
  z1_mimic_clean_fb_2xa100_1024env_20260716_044447/best_so_far.safetensors
```

For desktop deployment-style visualization, provide a matching ONNX policy and
its sibling `.meta.json` in an artifact directory such as:

```text
../checkpoints/Z1_UFO/compare_hand_vs_nohand/fakehand_32M
```

The current Z1 ONNX actor is robot-specific:

```text
input:  actor_obs [batch, 631]
output: action    [batch, 23]
```

The input is:

```text
actor_obs = state[52] + last_action[23] + history_actor[300] + z[256]
```

The `.meta.json` beside the ONNX is the authoritative joint-order and dimension
contract. Do not reuse this policy for another robot or a different Z1 action
layout without re-exporting a matching checkpoint.

## Framework Map

The Z1 pipeline follows this path:

```text
RobotState clips
  -> Z1 robot config + data manifest
  -> MJLab / MuJoCo-Warp vectorized training
  -> FBcprAuxAgent / FBcprAuxModel
  -> checkpoint + tracking evaluation
  -> backward encoder z trajectories
  -> ONNX actor export
  -> plain MuJoCo deploy and Tkinter slider tools
```

Detailed local notes:

- [Z1 ONNX Deploy Notes](tools/z1/README.md)
- [Z1 framework diagram](docs/diagrams/bfm_z1_training_framework.svg)
- [Z1 ONNX sim-to-sim notes](docs/z1_onnx_sim2sim_deploy_notes.md)
- [Z1 mimic semantic groups](docs/z1_mimic_semantic_groups.md)

## Running the Base Project

This workstream still uses the upstream UFO package layout and commands.

Install:

```bash
uv sync
```

Useful checks:

```bash
uv run ruff check .
uv run python -m compileall -q humanoidverse tests
uv run python tests/test_motion_data_adapter.py
```

Z1 smoke training:

```bash
./run_train.sh \
  --agent fb \
  --robot-config configs/robots/z1_23dof.yaml \
  --data-manifest configs/data/z1_mimic_clean.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_z1
```

## Credit and Upstream

This repository builds directly on RoboParty Lab's UFO project. Please cite and
link the original work when using this codebase:

- Upstream repository: <https://github.com/Roboparty/UFO>
- Project page: <https://roboparty.github.io/UFO/>
- Paper PDF: <https://roboparty.github.io/UFO/assets/UFO.pdf>
- Deploy branch: <https://github.com/Roboparty/UFO/tree/deploy>

The original UFO project is an unsupervised reinforcement learning framework for
humanoid control. Its `main` branch provides MJLab training, RobotState import,
tracking/goal/reward inference, and ONNX export. Its most complete upstream path
is Unitree G1; this repository extends that path for Magicbot Z1 experiments.

```bibtex
@misc{ufo2026,
  author       = {{RoboParty Lab Team}},
  title        = {UFO: An Unsupervised Reinforcement Learning Framework for Humanoid Control},
  year         = {2026},
  howpublished = {\url{https://github.com/Roboparty/UFO}},
  note         = {Project page: \url{https://roboparty.github.io/UFO/}}
}
```

License: see [LICENSE](LICENSE).
