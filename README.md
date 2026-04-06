# Humanoid Roller Skating

Roller-only RL training project for a Unitree G1 humanoid with inline skates, built on top of `mjlab` and a vendored `rsl_rl` backend.

## Workflow

The repo exposes one task and three primary workflows:

- train a policy: `train Mjlab-Roller-Flat-Unitree-G1`
- play a trained checkpoint: `play Mjlab-Roller-Flat-Unitree-G1 ...`
- build or validate AMP clips: `build-amp-dataset ...`, `validate-amp-dataset ...`

The canonical task is `Mjlab-Roller-Flat-Unitree-G1`. There is no extra skateboard or skater compatibility task in the cleaned-up surface.

## Repo Structure

- `src/mjlab_roller`: project package
- `src/mjlab_roller/cli`: user-facing entrypoints
- `src/mjlab_roller/tasks`: task registration and task configs
- `src/mjlab_roller/envs`: custom MuJoCo roller environment
- `src/mjlab_roller/assets`: robot asset and MuJoCo XML ownership
- `src/mjlab_roller/data`: AMP dataset contract and helpers
- `src/mjlab_roller/core`: shared control-space and project-path utilities
- `dataset`: reference poses and AMP clips
- `rsl_rl`: vendored RL backend
- `tests`: lightweight regression tests
- `docs/WORKFLOW.md`: step-by-step workflow and file map

## Training Model

- robot: Unitree G1 configured as a 23-DoF roller-skating system
- environment: manager-based MuJoCo RL environment
- control: joint-position policy over the controlled joint set
- reward structure: push, glide, steer, and transition phases plus regularization
- default learning method: PPO
- AMP code remains in the repo, but the default task configuration now trains without AMP

## AMP Dataset Contract

The runtime AMP loader consumes `.npy` clips stored under `dataset/roller_push`.

- each clip is a `[T, 23]` `float32` array
- columns follow the controlled joint order defined in [control_spec.py](src/mjlab_roller/core/control_spec.py)
- checked-in clips are bootstrap-quality projections from older data and should be replaced with retargeted roller-skating demonstrations for serious training

This repo does not include the raw video-to-mocap stage. It expects retargeted joint trajectories as input to `build-amp-dataset from-retargeted-poses`.

## Local Validation

The repo was previously validated with:

- `python -m compileall src rsl_rl tests`
- `PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"`
- `PYTHONPATH=src python -m mjlab_roller.cli.validate_amp_dataset --dataset-dir dataset/roller_push --num-frames 5`

## Known Limits

- the checked-in AMP clips are still bootstrap data, not publication-quality demonstrations
- runtime smoke testing still depends on having the `mjlab` stack installed locally
- reward weights and skate/contact physics are expected to need tuning once runtime validation is available
