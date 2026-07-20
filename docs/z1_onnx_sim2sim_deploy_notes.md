# Z1 ONNX Sim-to-Sim and Deploy Notes

This note summarizes how to use the exported Z1 UFO ONNX policy for MuJoCo
sim-to-sim, RK3588 deployment, and finite-state motion switching.

## Current Sim-to-Sim Artifact

The current local sim-to-sim smoke test runs the exported Z1 actor ONNX in plain
MuJoCo:

```text
z.pkl + robot proprioception -> actor_obs[631] -> ONNX actor -> action[23] -> PD target -> MuJoCo torque motors
```

Runner:

```powershell
UFO\.venv\Scripts\python.exe UFO\tools\z1\deploy_onnx_mujoco.py --clips aini kick cekongfan mabu
```

Terrain selection:

```powershell
UFO\.venv\Scripts\python.exe UFO\tools\z1\deploy_onnx_mujoco.py --terrain plane
UFO\.venv\Scripts\python.exe UFO\tools\z1\deploy_onnx_mujoco.py --terrain gravel
```

`plane` uses only the floor already present in the Z1 MJCF. `gravel` removes
that floor and injects a single hfield, avoiding simultaneous floor and gravel
contact geometry.

Outputs:

```text
checkpoints/Z1_UFO/compare_hand_vs_nohand/fakehand_32M/mujoco_sim2sim/
```

Generated videos:

- `aini_mujoco_onnx.mp4`
- `kick_mujoco_onnx.mp4`
- `cekongfan_mujoco_onnx.mp4`
- `mabu_mujoco_onnx.mp4`
- `mid_montage.png`
- `metrics.json`

## Real-Time Tkinter FSM Viewer

The interactive MuJoCo FSM viewer is:

```powershell
UFO\.venv\Scripts\python.exe UFO\tools\z1\ufo_fsm_tk_mujoco.py
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

For desktop debugging only, add:

```powershell
UFO\.venv\Scripts\python.exe UFO\tools\z1\ufo_fsm_tk_mujoco.py --reset-on-switch
```

The `--reset-on-switch` option teleports the MuJoCo state to the target clip
start when switching. This is useful because each motion starts from the same
initial condition used by its latent sequence, but it is not a real transition.
For sim-to-real or continuous sim-to-sim testing, leave that option disabled and
use guarded transitions through a stand/recover state instead.

The viewer does not use `z = 0` as idle. The current temporary idle holds the
first pose from `mabu`:

```powershell
UFO\.venv\Scripts\python.exe UFO\tools\z1\ufo_fsm_tk_mujoco.py --idle-source mabu
```

This is only a practical desktop stand-in. A zero latent is not guaranteed to be
a valid behavior unless the policy was explicitly trained so that zero maps to
idle. For deployment, add a curated idle/stand mocap clip or train an AMP/UFO
stand latent and make that the FSM's `IDLE/STAND` state.

## ONNX Runtime Contract

The actor ONNX is:

```text
input:  actor_obs [batch, 631]
output: action    [batch, 23]
```

The input vector is:

```text
actor_obs = state[52] + last_action[23] + history_actor[300] + z[256]
```

Where:

```text
state = dof_pos_rel[23] + dof_vel[23] + projected_gravity[3] + base_ang_vel[3]
```

For this Z1 policy:

- `dof_pos_rel = dof_pos - default_dof_pos`
- `base_ang_vel` is scaled by `0.25`
- `projected_gravity` is gravity expressed in the base frame
- `last_action` is the previous normalized/scaled policy action
- `history_actor` is a fixed history buffer
- `z` is the 256-D skill/motion latent

The controlled joint order is recorded in:

```text
FBcprAuxModel_41613312.meta.json
```

Do not change this order during deployment.

## Action Conversion

The ONNX actor outputs a 23-D normalized action. The deployed controller converts
this to PD targets.

For this Z1 config:

```text
scaled_action = clip(policy_action * 5.0, -5.0, 5.0)
action_target_scale[j] = 0.25 * effort_limit[j] / kp[j]
q_target[j] = default_q[j] + scaled_action[j] * action_target_scale[j]
```

The MuJoCo sim-to-sim runner then applies:

```text
torque[j] = kp[j] * (q_target[j] - q[j]) - kd[j] * qd[j]
torque[j] = clip(torque[j], -effort_limit[j], effort_limit[j])
```

This is the same deploy-facing interpretation that should be replicated on the
real robot side.

## RK3588 / SimuRLacra Deployment Considerations

Deploying on RK3588 is less about the neural-network math and more about runtime
determinism.

The important differences from desktop MuJoCo are:

- inference backend: ONNX Runtime CPU, RKNN, MNN, NCNN, or another ARM backend
- control frequency: current policy loop is 50 Hz
- latency: sensor read -> observation build -> inference -> command send -> actuator response
- jitter: variable delay is often worse than fixed delay
- observation freshness: IMU and joint states should be sampled consistently
- action hold: low-level PD should hold the last target between policy ticks
- safety clamps: joint limits, velocity limits, torque limits, fall detection, emergency stop

Measure these on the board:

```text
observation build time
policy inference time
command publish time
actual loop period
loop jitter
sensor timestamp age
```

A reasonable first target is:

```text
policy frequency: 50 Hz
loop period:      20 ms
inference time:   preferably < 5 ms
```

If total latency approaches one or two full policy periods, evaluate with
explicit delay compensation or retrain/evaluate with observation/action delay.

## FSM Motion Switching

For UFO, the policy actor is shared. The motion identity is mostly represented by
the latent `z`.

A minimal finite-state machine can switch the `z` source:

```text
IDLE
  -> AINI
  -> KICK
  -> CEKONGFAN
  -> MABU
  -> RECOVER
```

Each state should own:

```text
z_sequence
frame_index
entry condition
exit condition
blend duration
safety condition
```

The runtime loop is:

```python
state = fsm.current_state
z = state.get_z(t)

actor_obs = concat(state_obs, last_action, history_actor, z)
action = policy(actor_obs)
send_pd(action)

fsm.update(robot_state, command, safety_flags)
```

Do not abruptly jump between unrelated `z` vectors. Blend them:

```python
z_blend = (1.0 - alpha) * z_old + alpha * z_new
z_blend = normalize(z_blend) * sqrt(256)
```

Use a short blend window, for example:

```text
0.2 s to 0.5 s
```

Also gate transitions by robot state. For example, entering `kick` from a deep
`mabu` pose may be unsafe unless the robot first returns to a compatible stand
or ready pose.

Recommended initial transition policy:

```text
IDLE/STAND can enter any skill
KICK returns to IDLE/STAND
MABU returns to IDLE/STAND
AINI can loop or return to IDLE/STAND
CEKONGFAN needs stricter safety gating
```

## Traditional FSM vs UFO Latent FSM

Traditional robot controllers often switch between separate controllers:

```text
stand controller -> walk controller -> kick controller -> balance controller
```

The transition motion is added because each controller expects a particular
initial condition. Balance or walking states are explicit bridge states.

With UFO, the actor is shared:

```text
same actor + different z
```

Instead of:

```text
different controller per state
```

This makes motion switching more compact, but it does not remove the need for
transition management. If the robot is in an incompatible pose and the runtime
switches to a very different `z`, the actor can saturate actions or lose balance.

The practical difference is:

```text
Traditional FSM:
  transition is an explicit motion/controller between states.

UFO latent FSM:
  transition can be a z blend, a stand/recover z, or a guarded switch at
  compatible robot states.
```

For real robot deployment, keep a conservative outer safety FSM:

```text
BOOT -> STAND -> SKILL -> RECOVER -> E_STOP
```

Inside `SKILL`, use latent switching:

```text
skill_z = aini / kick / cekongfan / mabu / ...
```

This combines the flexibility of UFO latent control with a safety structure that
is easier to reason about on hardware.
