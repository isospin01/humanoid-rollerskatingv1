# Humanoid Roller Skating

Roller-only training project for a Unitree G1 humanoid with inline skates, built on top of `mjlab` and a vendored `rsl_rl` AMP/PPO stack.

## Current Status

The repository is in a cleaned-up roller-only state.

- supported workflows: `train`, `play`
- canonical task: `Mjlab-Roller-Flat-Unitree-G1`
- compatibility alias: `Mjlab-Skater-Flat-Unitree-G1`
- main environment: custom MuJoCo manager-based RL environment for a 23-DoF G1 roller setup
- training method: single shared actor-critic policy trained with PPO plus an AMP discriminator

## What Is Implemented

- roller asset ownership under `src/mjlab_husky/asset_zoo/robots/roller`
- task bootstrap and registry wiring for the roller task
- custom skating environment with phase-based rewards across push, glide, steer, and transition segments
- strict AMP dataset contract using controlled-joint clips with shape `[T, 23]`
- dataset tools:
  - `build-amp-dataset`
  - `validate-amp-dataset`
- training and play entrypoints:
  - `train`
  - `play`
- lightweight tests for task bootstrap, project-relative path resolution, and AMP dataset validation

## Data Pipeline

The runtime AMP loader consumes `.npy` clips stored under `dataset/roller_push`.

- each clip is a `[T, 23]` float32 array
- columns follow the controlled joint order defined in `src/mjlab_husky/control_spec.py`
- checked-in clips are currently bootstrap projections from legacy data
- the repo also supports rebuilding AMP clips from retargeted joint-position arrays via `build-amp-dataset from-retargeted-poses`

Important: this repository does not contain the raw video-to-mocap extraction stage. It expects retargeted joint trajectories as input to the dataset builder.

## Verified In This Workspace

The following checks were reported as passing locally during cleanup:

- `python -m compileall src rsl_rl tests`
- `PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"`
- `PYTHONPATH=src python -m mjlab_husky.scripts.validate_amp_dataset --dataset-dir dataset/roller_push --num-frames 5`

## Known Gaps

- `mjlab` is not installed in this workspace, so runtime simulator smoke tests are still unverified here
- checked-in AMP clips are still bootstrap-quality, not publication-quality retargeted demonstrations
- physics and reward tuning are still expected after runtime validation:
  - wheel friction
  - scrape threshold
  - lateral slip threshold
  - skate mass and inertia
  - reward balance across skating phases

## Layout

- `src/mjlab_husky`: project code
- `src/mjlab_husky/envs`: custom roller environment
- `src/mjlab_husky/tasks`: task registration and config
- `src/mjlab_husky/scripts`: CLI entrypoints
- `dataset`: reference poses and AMP clips
- `rsl_rl`: vendored RL backend
- `tests`: lightweight regression tests
