# Roller Project Status

Last updated: 2026-03-16

## Current State

The repository is now a roller-only training project with the planned cleanup and code fixes applied.

- supported workflows: `train`, `play`
- canonical task: `Mjlab-Roller-Flat-Unitree-G1`
- compatibility alias: `Mjlab-Skater-Flat-Unitree-G1`
- removed from the repo surface: legacy skateboard environment code, duplicate `skate_push` data, the old `test_scene` evaluation path, and stale project documentation

## Verified In This Workspace

The following checks pass locally:

- `python -m compileall src rsl_rl tests`
- `PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"`
- `PYTHONPATH=src python -m mjlab_husky.scripts.validate_amp_dataset --dataset-dir dataset/roller_push --num-frames 5`

## Implemented Fixes

- made the roller asset/config path self-owned under `asset_zoo/robots/roller`
- fixed `train.py` task bootstrapping so the task registry is populated before CLI parsing
- made reference-pose and AMP dataset paths resolve from the project root instead of the current working directory
- replaced the old AMP loader with a strict controlled-joint loader that expects `[T, 23]` clips and validates them before training
- added `build-amp-dataset` and `validate-amp-dataset` CLIs for regenerating AMP clips from retargeted pose sequences
- regenerated the checked-in AMP clips into the new contract and added `dataset/roller_push/manifest.json`
- improved `play --viewer auto` so native viewer selection works better off Linux
- added lightweight tests for path resolution, AMP dataset validation, and task bootstrap wiring

## Remaining External Work

### 1. Runtime smoke testing is still blocked on dependencies

`mjlab` is not installed in this workspace, so these runtime checks are still unverified here:

- environment construction
- simulator startup
- short `train` smoke run
- `play` execution with a checkpoint
- contact and wheel behavior under MuJoCo runtime

### 2. Checked-in AMP clips are still bootstrap approximations

The repository now has a clean AMP data contract and a builder for retargeted pose sequences, but the included clips were projected from the previous legacy dataset because no licensed human roller-skating source footage was present locally.

Treat the current clips as development bootstrap data, not publication-quality demonstrations.

### 3. Physics tuning remains a training-time task

After runtime validation is available, tuning is still expected for:

- wheel friction
- scrape threshold
- lateral slip threshold
- inline-skate mass and inertia
- reward weights across push, glide, steer, and transition phases


## Status Summary

The code and repo issues from the implementation plan are addressed. The remaining work is external to the code changes already made: install the runtime stack, run simulator smoke tests, replace the bootstrap AMP clips with real retargeted roller-skating demonstrations, and remove the two locked temp directories once Windows allows it.
