# Z1 Recovery Motion Data

`train_npz/` contains the immutable raw fall-and-get-up candidate clips. They use the source schema with `joint_pos`, `body_pos_w`, and `body_quat_w`; UFO training cannot load this schema directly.

`robot_state_npz/` is generated output for UFO. Each file contains `root_pos`, `root_quat`, `dof_pos`, `fps`, `joint_names`, and conversion provenance. Do not edit generated files by hand or overwrite the raw source clips.

Run the converter from the repository root:

```powershell
uv run python tools/z1/convert_z1_recovery_to_robot_state_npz.py --force
```

The converter finds the `pelvis` through `body_names`, maps `joint_pos` to `dof_pos`, normalizes pelvis quaternions in `[x, y, z, w]` order, and applies one clip-wide ground offset based on the minimum body height.

The converted files are candidates, not automatically approved training data. Complete mechanical and MuJoCo visual review before adding them to the recovery dataset of the FB training manifest. Recovery data is auxiliary training data only; the deployment manifest must continue to contain the formal 20 tasks and must not reference this directory.
