# Smoke Test

This is the minimum runtime validation checklist for the current PPO-only setup.

Goal:

- prove the local Python environment can import the project and `mjlab`
- prove the roller environment can construct and step
- prove a short PPO training run starts without immediate runtime failures

This is not a learning-quality test. It only checks that the runtime path is healthy enough to begin experiments.

## 1. Preconditions

Use the repo root:

```powershell
cd C:\Users\xumuc\Desktop\humanoid-rollerskatingv1-main
```

The project currently expects Python `>=3.12,<3.14`.

If you are running from source without installing the package, set:

```powershell
$env:PYTHONPATH="src"
```

If you prefer an installed workflow, install the repo first and then you can skip `PYTHONPATH`.

## 2. Basic Import Check

Run:

```powershell
$env:PYTHONPATH="src"
python -c "import mjlab; import mjlab_roller; print('imports ok')"
```

Expected result:

- the command prints `imports ok`

If this fails:

- `No module named 'mjlab'`: the runtime stack is not installed yet
- `No module named 'mjlab_roller'`: you forgot `PYTHONPATH=src` or the package is not installed

## 3. Task Bootstrap Check

Run:

```powershell
$env:PYTHONPATH="src"
python -c "from mjlab_roller.tasks.bootstrap import bootstrap_task_registry; bootstrap_task_registry(); from mjlab_roller.tasks.registry import list_tasks; print(list_tasks())"
```

Expected result:

- the output includes `Mjlab-Roller-Flat-Unitree-G1`

## 4. Viewer Smoke Test With Dummy Policy

This is the fastest useful simulator check.

Run:

```powershell
$env:PYTHONPATH="src"
python -m mjlab_roller.cli.play Mjlab-Roller-Flat-Unitree-G1 --agent zero --num-envs 1 --viewer native
```

What this checks:

- environment construction
- MuJoCo model load
- sensor setup
- viewer startup
- stepping loop

Expected result:

- the viewer opens
- the robot appears in the scene
- the sim runs without immediate crash

Notes:

- `--agent zero` is not meaningful behavior; it is only for runtime validation
- on Windows, `native` is the right first choice

If you want a second quick check:

```powershell
$env:PYTHONPATH="src"
python -m mjlab_roller.cli.play Mjlab-Roller-Flat-Unitree-G1 --agent random --num-envs 1 --viewer native
```

## 5. PPO Training Smoke Test

Run a very small training job first:

```powershell
$env:PYTHONPATH="src"
python -m mjlab_roller.cli.train Mjlab-Roller-Flat-Unitree-G1 --env.scene.num_envs 8 --agent.max_iterations 2
```

What this checks:

- training entrypoint
- env reset and rollout collection
- PPO forward and backward pass
- logging directory creation
- checkpoint save path

Expected result:

- training starts
- at least 2 iterations complete
- logs appear under `logs/rsl_rl/g1_roller_ppo`

## 6. Slightly Larger Sanity Run

If the 2-iteration test passes, try:

```powershell
$env:PYTHONPATH="src"
python -m mjlab_roller.cli.train Mjlab-Roller-Flat-Unitree-G1 --env.scene.num_envs 32 --agent.max_iterations 10
```

You are looking for:

- no NaNs
- no sensor/shape/runtime exceptions
- stable reward logging
- regular checkpoint saves

## 7. Optional Stability Guard

If the simulator becomes unstable, retry with:

```powershell
$env:PYTHONPATH="src"
python -m mjlab_roller.cli.train Mjlab-Roller-Flat-Unitree-G1 --env.scene.num_envs 8 --agent.max_iterations 2 --enable-nan-guard True
```

## 8. Pass Criteria

I would call the project runtime-ready for training only if all three pass:

1. import check
2. dummy `play` viewer check
3. 2-iteration PPO smoke test

That does not mean the policy will learn good skating yet.

It only means:

- the code path is wired correctly enough to begin training experiments
- remaining problems are more likely to be reward design, contact tuning, or learning stability rather than basic runtime breakage

## 9. Likely Failure Points

The most likely problems when you first run this on a new machine are:

- `mjlab` not installed
- missing GPU / EGL / viewer dependencies
- MuJoCo runtime mismatch
- sensor name mismatch in the G1 XML
- instability from contact or friction settings
- insufficient memory when `num_envs` is too high

## 10. Recommended First Command

If you only run one command first, use this:

```powershell
$env:PYTHONPATH="src"
python -m mjlab_roller.cli.play Mjlab-Roller-Flat-Unitree-G1 --agent zero --num-envs 1 --viewer native
```

If that works, move to the 2-iteration PPO smoke test.
