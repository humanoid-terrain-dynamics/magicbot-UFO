# Z1 UFO ONNX Deploy Notes

This folder contains the local Z1 UFO sim-to-sim and FSM deployment helpers.

Main files:

- `deploy_onnx_mujoco.py`: batch ONNX rollout in plain MuJoCo, used to generate
  per-skill videos and metrics.
- `ufo_fsm_tk_mujoco.py`: Tkinter-hosted real-time MuJoCo viewer for switching
  UFO latent states.
- `ufo_44_slider_tk_mujoco.py`: Tkinter-hosted real-time MuJoCo viewer for the
  full 44-motion curated prior, selected with one slider.
- `generate_z1_44_latents.py`: builds the official `z` cache for the 44-motion
  slider viewer.
- `ufo_fsm_mujoco_viewer.py`: older passive MuJoCo viewer.
- `vis_npz_dataset_labeled.py`: NPZ/mocap browser used as the Tkinter UI
  reference.

## Current Artifacts

The current exported policy artifacts are expected under:

```text
../checkpoints/Z1_UFO/compare_hand_vs_nohand/fakehand_32M
```

Important files:

- `FBcprAuxModel_41613312.onnx`: actor policy
- `FBcprAuxModel_41613312.meta.json`: controlled joint order
- `aini_zs.pkl`, `kick_zs.pkl`, `cekongfan_zs.pkl`, `mabu_zs.pkl`: latent
  sequences for the four test skills

## Batch Sim-to-Sim

From the repository parent folder:

```powershell
UFO\.venv\Scripts\python.exe UFO\tools\z1\deploy_onnx_mujoco.py --clips aini kick cekongfan mabu
```

Outputs are written to:

```text
checkpoints/Z1_UFO/compare_hand_vs_nohand/fakehand_32M/mujoco_sim2sim
```

## Real-Time Tkinter FSM Viewer

Continuous switching mode:

```powershell
UFO\.venv\Scripts\python.exe UFO\tools\z1\ufo_fsm_tk_mujoco.py
```

Debug mode that teleports to each clip start on switch:

```powershell
UFO\.venv\Scripts\python.exe UFO\tools\z1\ufo_fsm_tk_mujoco.py --reset-on-switch
```

Controls:

- `0`: IDLE/STAND
- `1`: `aini`
- `2`: `kick`
- `3`: `cekongfan`
- `4`: `mabu`
- `R`: reset current pose
- `Space`: pause/resume
- `Q` or `Esc`: quit

`--reset-on-switch` is useful for checking whether each latent can play from its
own source initial condition. It is not a real transition and should not be used
as evidence that the robot can move from one arbitrary state into another.

## Policy Observation Contract

The ONNX actor uses:

```text
input:  actor_obs [batch, 631]
output: action    [batch, 23]
```

The input vector is:

```text
actor_obs = state[52] + last_action[23] + history_actor[300] + z[256]
```

where:

```text
state = dof_pos_rel[23] + dof_vel[23] + projected_gravity[3] + base_ang_vel[3]
```

The controlled joint order must match `FBcprAuxModel_41613312.meta.json`.

## Official PD Convention

The referenced local file `AAAtmp/PD_from_mimic/z1.py` is currently empty, but
the Z1 configs in this repo state that the PD/default pose values were copied
from `magicbot-mimic whole_body_tracking/robots/z1.py`.

Current control convention:

```text
control_type: P
action_scale: 0.25
action_rescale: true
normalize_action_to: 5.0
clip_torques: true
```

The deploy runner mirrors the training adapter in
`humanoidverse/agents/envs/humanoidverse_mjlab.py`: when `action_rescale` is
enabled, action target scale is:

```text
action_target_scale[j] = action_scale * effort_limit[j] / kp[j]
```

The deployed torque computation is:

```python
action_scaled = clip(policy_action * 5.0, -5.0, 5.0)
q_target = default_q + action_scaled * (0.25 * effort_limit / kp)
torque = kp * (q_target - q) - kd * qd
torque = clip(torque, -effort_limit, effort_limit)
```

This means the actor outputs normalized joint target offsets, not direct torque.
The PD layer converts those offsets into clipped motor torques.

## Latent Switching

UFO uses one actor and changes motion behavior through the latent `z`.

Direct switching would be:

```text
frame t:     z = z_old
frame t + 1: z = z_new
```

The viewer instead blends for a short window:

```python
z_blend = (1.0 - alpha) * z_old + alpha * z_new
z_blend = normalize(z_blend) * sqrt(256)
```

This is what:

```text
z_old -> blended z -> z_new
```

means. It is a latent-space transition only. It does not guarantee a physically
valid bridge between two incompatible body states.

## FSM Deployment Policy

The deployed hardware FSM should not expose arbitrary direct transitions between
all skills.

Recommended structure:

```text
BOOT -> STAND -> SKILL -> RECOVER/STAND -> SKILL -> RECOVER/STAND -> E_STOP
```

Inside `SKILL`, the selected UFO latent can be:

```text
aini / kick / mabu / other deployable skill
```

The outer FSM should enforce:

- posture gate before entering a skill
- root-height and base-tilt safety checks
- joint-limit and torque-limit checks
- timeout or motion-end condition
- recovery path back to stand
- emergency stop

For desktop debugging, latent blending alone is acceptable. For sim-to-real,
motion A should normally return to a stable gate before motion B starts.

## Skill Clip Selection

For deployment, a skill should ideally contain:

```text
stable preparation -> main motion -> landing/recovery -> stable end
```

Partial clips can still be useful, but they should be marked differently:

- `deployable`: stable start and stable end, safe to expose in the FSM.
- `subskill`: only valid from a known compatible pose or previous state.
- `training/debug only`: useful for representation learning or visual checks,
  but not safe as a direct deploy state.

For example, if `cekongfan` only contains the middle part of the flip or lacks a
clean landing/recovery, do not expose it as a direct user-selectable deploy
state. Better options are:

- re-trim a complete `cekongfan` with preparation and recovery,
- add explicit entry and recovery clips,
- keep it as a subskill that can only be entered from a compatible pose,
- remove it from the deploy FSM while keeping it in the training/debug dataset.

Do not delete partial clips blindly. Label them and route them through the FSM
according to their entry and exit requirements.

## Idle / Stand

The current Tkinter viewer does not treat `z = 0` as a real idle skill. A zero
latent is not guaranteed to map to standing unless the policy was trained that
way.

The current temporary idle holds the first pose of `mabu`:

```powershell
UFO\.venv\Scripts\python.exe UFO\tools\z1\ufo_fsm_tk_mujoco.py --idle-source mabu
```

For hardware deployment, train or curate a real idle/stand skill and make that
the FSM's `STAND` state.

## 44-Motion Slider Viewer

The 44-motion prior comes from these four training-cache files:

```text
cache/motion_data/z1_mimic_clean/z1_cyclic_locomotion_train_near10s_ufo.pkl
cache/motion_data/z1_mimic_clean/z1_atomic_skills_train_near10s_ufo.pkl
cache/motion_data/z1_mimic_clean/z1_acrobatics_recovery_train_near10s_ufo.pkl
cache/motion_data/z1_mimic_clean/z1_pose_low_motion_train_near10s_ufo.pkl
```

Together they contain 44 curated motions. Before launching the slider viewer,
generate the matching official backward-encoder latents:

```powershell
UFO\.venv\Scripts\python.exe UFO\tools\z1\generate_z1_44_latents.py
```

This writes:

```text
checkpoints/Z1_UFO/compare_hand_vs_nohand/fakehand_32M/z44_latents/
```

Then launch:

```powershell
UFO\.venv\Scripts\python.exe UFO\tools\z1\ufo_44_slider_tk_mujoco.py
```

The slider selects motion index `0..43`. Left/Right nudge the slider by one
motion. The transition is still a latent blend; it is not a guaranteed physical
bridge between incompatible skills.
