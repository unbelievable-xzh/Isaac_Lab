# Isaac_Lab
very_good~

## Plan-C model-based lift task

This repository extends the Isaac Lab lift task with a model-based variant that
implements the "Plan C" idea from the accompanying design notes.  The new
environment and agent configuration are registered under the Gym identifier
`Isaac-Lift-Cube-Franka-v2` and can be launched through the usual Isaac Lab
entry points.  Key additions include:

- Noisy actor observations paired with Kalman-filtered critic inputs.
- An online TorchScript-capable dynamics ensemble that provides rollouts and
  adaptive Kalman parameters as auxiliary features.
- A recurrent PPO agent configuration tuned for the richer observation space.

To start training the model-based agent with RSL-RL, run:

```bash
isaaclab.sh -p source/standalone/workflows/rlgames/train.py --task Isaac-Lift-Cube-Franka-v2
```

The environment automatically maintains the dynamics ensemble, so no extra
setup is required beyond selecting the new task id.
