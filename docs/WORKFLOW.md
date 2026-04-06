# Workflow

This repository is organized around one roller task: `Mjlab-Roller-Flat-Unitree-G1`.

## Start Here

Read these files first if you want to understand the main workflow:

1. `src/mjlab_roller/cli/train.py`
2. `src/mjlab_roller/tasks/roller/config/g1/__init__.py`
3. `src/mjlab_roller/tasks/roller/config/g1/env_cfgs.py`
4. `src/mjlab_roller/tasks/roller/roller_env_cfg.py`
5. `src/mjlab_roller/envs/g1_roller_rl_env.py`
6. `src/mjlab_roller/tasks/roller/mdp/rewards.py`
7. `src/mjlab_roller/data/amp_dataset.py`

## Typical Workflow

1. Validate or rebuild the AMP dataset under `dataset/roller_push`.
2. Launch training with `train Mjlab-Roller-Flat-Unitree-G1`.
3. Inspect logs and checkpoints under `logs/rsl_rl/g1_roller_ppo`.
4. Load a checkpoint with `play Mjlab-Roller-Flat-Unitree-G1`.

## Package Map

- `src/mjlab_roller/cli`
  Commands for training, play, and AMP dataset maintenance.
- `src/mjlab_roller/core`
  Shared constants and project-root path handling.
- `src/mjlab_roller/data`
  AMP clip normalization, manifest handling, and validation.
- `src/mjlab_roller/assets`
  MuJoCo XML ownership and robot actuator configuration.
- `src/mjlab_roller/tasks`
  Task registry plus roller-specific environment and RL config wiring.
- `src/mjlab_roller/envs`
  Runtime environment implementation.
- `rsl_rl`
  Vendored PPO/AMP backend.
