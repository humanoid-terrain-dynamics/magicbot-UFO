1. Ordinary Sim2Sim
Terminal A, start MuJoCo:

cd "$UFO_ROOT"
conda activate ufo-deploy
python -m sim_env.base_sim \
  --robot_config ./config/robot/g1.yaml \
  --scene_config ./config/scene/g1_29dof.yaml
Terminal B, start the policy:

cd "$UFO_ROOT"
conda activate ufo-deploy
python rl_policy/ufo_policy.py \
  --robot_config config/robot/g1.yaml \
  --policy_config config/policy/g1_policy.yaml \
  --model_path model/g1_policy/exported/FBcprAuxModel.onnx \
  --task config/exp/tracking/tracking.yaml
Keyboard controls in the policy terminal:

i   interpolate to default standing pose
]   enable policy action
[   start tracking motion
p   reset tracking motion to stop frame
o   stop policy action and hold current joints
n   next reward/goal z for reward/goal tasks