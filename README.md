# MagicBot UFO Z1 Deployment

This branch contains the self-contained 23-DoF Z1 MuJoCo deployment bundle:
robot assets, simulation scripts, control contracts, the release manifest, and
tracking latents.

## Model Artifact

Large model inputs are deliberately excluded from this GitHub repository by
`.gitignore`. The complete staged Z1 policy artifact is published on Hugging
Face at [PhangHongHao/UFO-Z1](https://huggingface.co/datasets/PhangHongHao/UFO-Z1),
under the `z1_policy/` directory.

Restore it at this repository root before running the deployment:

```powershell
hf download PhangHongHao/UFO-Z1 --repo-type dataset --include "z1_policy/**" --local-dir .
```

The downloaded artifact includes the policy checkpoint, `policy.onnx`,
`backward_encoder.onnx`, tracking latents, the control contract, and the
SHA-256 release manifest. Do not hand-edit these release inputs.

## Smoke Test

With the project environment installed, run the required deterministic check
from the repository root:

```powershell
uv run python scripts/sim2sim_mujoco.py --headless --latent zs_0.pkl --max-steps 250
```

Review `outputs/sim2sim_metrics.json` after the run. Local sim-to-sim success
does not establish sim-to-real validity.
