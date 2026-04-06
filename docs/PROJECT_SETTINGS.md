# Project Settings Snapshot

This file is a plain-language snapshot of the key settings currently committed in the repo.

Source of truth:

- `src/mjlab_roller/tasks/roller/roller_env_cfg.py`
- `src/mjlab_roller/tasks/roller/config/g1/env_cfgs.py`
- `src/mjlab_roller/envs/g1_roller_rl_env.py`
- `src/mjlab_roller/tasks/roller/mdp/rewards.py`
- `src/mjlab_roller/tasks/roller/mdp/velocity_command.py`
- `src/mjlab_roller/tasks/roller/mdp/terminations.py`
- `src/mjlab_roller/tasks/roller/mdp/observations.py`
- `src/mjlab_roller/tasks/roller/config/g1/rl_cfg.py`
- `src/mjlab_roller/rl/config.py`
- `src/mjlab_roller/assets/robots/roller/g1.py`
- `rsl_rl/modules/actor_critic.py`
- `rsl_rl/runners/amp_on_policy_runner.py`
- `rsl_rl/modules/discriminator_multi.py`

## 1. Current Training Stack

- Task ID: `Mjlab-Roller-Flat-Unitree-G1`
- Robot: Unitree G1 roller-skating setup with 23 controlled joints
- Policy: feedforward actor-critic MLP
- RL algorithm: PPO
- Extra guidance: AMP is present in the repo, but disabled in the current default task config
- Curriculum: none
- Environment family: custom manager-based MuJoCo RL environment

Important note:

- The code currently trains one shared policy over the full skating cycle.
- It does not train each phase separately.
- The default task now uses pure PPO without AMP reward.

## 2. Core Time And Phase Structure

The environment uses a repeating skating cycle with a fixed cycle time of `6.0 s`.

Phase ratios:

- `0.00 -> 0.30`: push
- `0.30 -> 0.40`: push-to-glide transition
- `0.40 -> 0.65`: glide
- `0.65 -> 0.75`: glide-to-steer transition
- `0.75 -> 0.95`: steer
- `0.95 -> 1.00`: steer-to-push transition

Equivalent durations within one 6-second cycle:

- push: `1.8 s`
- push-to-glide: `0.6 s`
- glide: `1.5 s`
- glide-to-steer: `0.6 s`
- steer: `1.2 s`
- steer-to-push: `0.3 s`

How phase is used:

- The environment computes a normalized phase value in `[0, 1]`.
- That phase drives reward gating.
- The policy directly observes the scalar `phase`.
- The critic additionally observes the one-hot-like `contact_phase` vector.

Special case:

- If the commanded forward speed is below `0.1`, the environment marks the agent as `still`.
- In that case, the phase timer is reset to `0`, so the cycle does not progress while standing still.

## 3. Simulation And Environment Setup

### Simulation

- terrain: flat plane
- physics timestep: `0.005 s`
- decimation: `4`
- effective policy/control step: `0.02 s`
- physics rate: `200 Hz`
- policy rate: `50 Hz`
- training episode length: `20.0 s`
- training episode length in policy steps: `1000`
- viewer camera target: `torso_link`

MuJoCo / sim settings:

- `nconmax = 55`
- `njmax = 300` in the G1-specific config
- `iterations = 10`
- `ls_iterations = 20`
- `ccd_iterations = 50`
- `contact_sensor_maxmatch = 64`

### Robot Setup

- controlled joint count: `23`
- action type: joint position action
- `use_default_offset = True`
- the generic base action scale `0.5` is overridden by a per-joint actuator-derived scale in `src/mjlab_roller/assets/robots/roller/g1.py`
- robot initial state is a push-oriented keyframe (`PUSH_INIT_KEYFRAME`)
- soft joint position limit factor: `0.9`

### Sensors Used By The Task

- `left_skate_contact`
- `right_skate_contact`
- `left_boot_scrape`
- `right_boot_scrape`
- `robot_collision`
- `illegal_contact`

Contact sensor intent:

- skate contact sensors track wheel contact and air time
- boot scrape sensors detect frame scraping against terrain
- robot collision tracks self-collision under the pelvis subtree rule
- illegal contact tracks shin, brace, arm, wrist, hand, and pelvis contacts

## 4. Commands

The main command is `skate`.

Command settings:

- forward speed range: `0.0` to `1.5 m/s`
- heading command enabled: yes
- relative heading range: `-pi/4` to `+pi/4`
- `rel_heading_envs = 1.0`
- `rel_standing_envs = 0.0`
- resampling time range: `(20.0, 20.0)`

What this means in practice:

- the forward-speed command is sampled once per episode by default, because the episode length is also `20 s`
- heading targets are resampled again when the environment exits the `glide -> steer` transition
- the command observed by the policy is a 2D quantity: forward speed and heading target

## 5. How The Policy Is Guided By Time

The policy is not recurrent. It is guided by time and phase in four main ways:

### Explicit phase observation

- The actor receives a scalar `phase`.
- This gives the policy direct knowledge of where it is in the skating cycle.

### Short history window

- The policy observation group uses `history_length = 5`.
- Because the policy step is `0.02 s`, this gives the actor a `0.1 s` rolling window of recent observations.
- This is the main short-term temporal context for the feedforward policy.

### Phase-gated rewards

- Push rewards are active only in push.
- Glide rewards are active only in glide.
- Steer rewards are active only in steer.
- Transition rewards are active only during transition segments.
- Regularization rewards are active throughout.

### Transition targets

- During the three transition phases, the environment creates body-position and body-orientation targets.
- Position targets use a Bezier interpolation.
- Orientation targets use quaternion slerp.
- These targets are then fed to the critic and used by the transition rewards.

Additional current behavior:

- `steer_phase_mask = steer_phase`
- `transition_active_mask` is true during all three transitions

Note:

- The environment still computes `amp_active_mask = push_phase`, but the default training configuration no longer uses the AMP runner.

## 6. Observations

### Policy Observations

The policy group contains:

- `command`
- `heading`
- `base_ang_vel`
- `projected_gravity`
- `joint_pos`
- `joint_vel`
- `actions`
- `phase`

Noise/corruption:

- policy corruption enabled: yes during training
- policy corruption disabled: yes during play mode
- base angular velocity noise: uniform `[-0.2, 0.2]`
- projected gravity noise: uniform `[-0.05, 0.05]`
- joint position noise: uniform `[-0.01, 0.01]`
- joint velocity noise: uniform `[-1.5, 1.5]`

Scales:

- command scale: `(2.0, 1.0)`
- heading scale: `1 / pi`
- base angular velocity scale: `0.25`
- joint velocity scale: `0.05`

Derived policy observation size:

- single-step actor observation width: `79`
- with history length `5` and flattened history: `395`

This `79` comes from:

- command: `2`
- heading: `1`
- base angular velocity: `3`
- projected gravity: `3`
- relative joint positions: `23`
- joint velocities: `23`
- last actions: `23`
- phase: `1`

### Critic Observations

The critic sees everything the actor sees, plus:

- `base_lin_vel`
- `heading_error`
- `left_skate_vel_local`
- `right_skate_vel_local`
- `left_skate_ang_vel_local`
- `right_skate_ang_vel_local`
- `skate_lean`
- `skate_marker_sep`
- `trans_target_pos_b`
- `trans_target_quat_b`
- `left_skate_forces`
- `right_skate_forces`
- `left_boot_scrape_force`
- `right_boot_scrape_force`
- `wheel_contact_summary`
- `contact_phase`

Derived critic observation size:

- current committed critic width: `242`

That `242` assumes the current sensor flattening used by the code:

- skate contact forces: `4 slots x 3 force values = 12` per skate
- boot scrape force: `1 slot x 3 force values = 3` per boot
- wheel contact summary: `8`
- contact phase: `6`
- transition target positions: `14 bodies x 3 = 42`
- transition target quaternions: `14 bodies x 4 = 56`

## 7. Reward Functions By Phase

All reward names and weights below are current committed values.

### Push Phase

Active only when the cycle is in push.

| Reward | Weight | Main idea |
|---|---:|---|
| `push_skate_lin_vel` | `3.0` | match commanded forward speed while suppressing lateral and vertical body velocity |
| `push_yaw_align` | `1.0` | align heading with target heading |
| `push_single_support_air_time` | `2.0` | encourage one skate to spend a bounded amount of time in the air |
| `push_stance_skate_contact` | `1.0` | require at least one skate in contact and no boot scrape |

Detailed settings:

- `push_skate_lin_vel`: `std = sqrt(0.25)`
- `push_yaw_align`: `std = sqrt(0.1)`
- `push_single_support_air_time`:
  - `threshold_min = 0.05`
  - `threshold_max = 0.35`
  - `command_threshold = 0.1`

### Glide Phase

Active only when the cycle is in glide.

| Reward | Weight | Main idea |
|---|---:|---|
| `glide_dual_skate_contact` | `2.5` | both skates in contact, no scrape |
| `glide_speed_retention` | `2.0` | keep commanded forward speed |
| `glide_skate_spacing` | `1.5` | keep left/right skate spacing near a target offset |
| `glide_lateral_stability` | `1.0` | reduce lateral skate velocity |

Detailed settings:

- `glide_speed_retention`: `std = sqrt(0.2)`
- `glide_skate_spacing`: `std = sqrt(0.05)`, target separation is `[0.0, -0.22, 0.0]`
- `glide_lateral_stability`: `std = sqrt(0.03)`

### Steer Phase

Active only when the cycle is in steer.

| Reward | Weight | Main idea |
|---|---:|---|
| `steer_dual_skate_contact` | `2.0` | both skates in contact, no scrape |
| `steer_joint_pos` | `1.5` | keep joint configuration near the steer reference pose |
| `steer_track_heading` | `5.0` | track heading target strongly |
| `steer_lean_guide` | `4.0` | lean the skates appropriately for the turn |

Detailed settings:

- `steer_joint_pos`: `std = sqrt(0.2)`
- `steer_track_heading`: `std = sqrt(0.02)`
- `steer_lean_guide`: `std = sqrt(0.02)`

### Transition Phases

These rewards are active during:

- push-to-glide
- glide-to-steer
- steer-to-push

| Reward | Weight | Main idea |
|---|---:|---|
| `transition_body_pos_tracking` | `8.0` | track Bezier-interpolated body position targets |
| `transition_body_rot_tracking` | `8.0` | track slerped body orientation targets |
| `transition_penalty_contact` | `-0.5` | penalize contact during transition |

Detailed settings:

- `transition_body_pos_tracking`: `std = sqrt(0.05)`
- `transition_body_rot_tracking`: `std = sqrt(0.1)`

### Regularization Rewards

These are always active.

| Reward | Weight | Main idea |
|---|---:|---|
| `reg_wheel_contact_coverage` | `0.5` | encourage broad wheel contact coverage |
| `dof_pos_limits` | `-5.0` | penalize soft joint limit violations |
| `action_rate_l2` | `-0.1` | penalize rapid action changes |
| `action_acc_l2` | `-0.1` | penalize action acceleration |
| `joint_vel_l2` | `-1e-3` | penalize joint speed |
| `joint_acc_l2` | `-2.5e-7` | penalize joint acceleration |
| `joint_torques_l2` | `-1e-6` | penalize torque magnitude |
| `self_collisions` | `-10.0` | penalize self-collision |
| `lean_flat` | `2.0` | keep skates flatter outside steer phase |
| `stand_still` | `1.0` | reward default-pose standing when commanded speed is very low |
| `boot_scrape_penalty` | `-1.0` | penalize frame scraping |

Detailed settings:

- `lean_flat`: `std = sqrt(0.03)`
- `stand_still`: `std = sqrt(0.1)`

## 8. Reset Events And Domain Randomization

Events:

- `push_robot`: interval disturbance every `5.0` to `10.0 s`
- `reset_robot_joints`: random controlled-joint reset offset in `[-0.01, 0.01]`
- `base_com`: random torso COM shift
- `left_skate_com`: random left skate COM shift
- `right_skate_com`: random right skate COM shift
- `robot_friction`: random global robot geom friction scale
- `skate_frame_friction`: absolute randomization of frame friction
- `wheel_friction`: wheel friction scaling on axes `[0, 2]`

Current randomization ranges:

- torso COM: `x,y in [-0.025, 0.025]`, `z in [-0.03, 0.03]`
- each skate COM:
  - `x in [-0.015, 0.015]`
  - `y in [-0.01, 0.01]`
  - `z in [-0.01, 0.01]`
- robot friction scale: `[0.5, 1.5]`
- skate frame friction absolute value: `[0.4, 1.2]`
- wheel friction scale: `[0.6, 1.4]`
- push disturbance velocity:
  - `x in [-0.5, 0.5]`
  - `y in [-0.25, 0.25]`

Play-mode differences:

- episode length becomes `60 s`
- policy corruption is disabled
- only `time_out` termination remains
- `push_robot` disturbance is removed

## 9. Terminations

Training-mode termination conditions:

- `time_out`
- `fell_over`: bad orientation beyond `70 degrees`
- `bad_skate_contact_loss`: both skates lose contact
- `excessive_boot_scrape`: any boot scrape time exceeds `0.2 s`
- `excessive_lateral_slip`: max local lateral wheel/body slip exceeds `0.8`
- `illegal_contact`

## 10. PPO Policy And Optimizer Hyperparameters

### Actor-Critic Network

Current policy config:

- actor hidden dims: `(512, 256, 128)`
- critic hidden dims: `(512, 256, 128)`
- activation: `ELU`
- actor observation normalization: enabled
- critic observation normalization: enabled
- initial action noise std: `1.0`
- noise std type: `scalar`
- recurrent policy: no

Model structure:

- actor: MLP from flattened actor observations to action means
- critic: separate MLP from critic observations to scalar value
- action distribution: Gaussian with learned global standard deviation vector

### PPO

Current PPO settings:

- learning rate: `1e-3`
- schedule: `adaptive`
- desired KL: `0.01`
- clip parameter: `0.2`
- gamma: `0.99`
- lambda (GAE): `0.95`
- entropy coefficient: `0.005`
- value loss coefficient: `1.0`
- clipped value loss: enabled
- max grad norm: `1.0`
- learning epochs per update: `5`
- mini-batches per update: `4`
- normalize advantage per mini-batch: disabled

Runner-level defaults in the committed task config:

- experiment name: `g1_roller_ppo`
- save interval: every `500` iterations
- steps per env per rollout: `24`
- max iterations: `50_000`
- default seed: `42`
- default logger: `tensorboard`

Derived rollout facts:

- one rollout per env covers `24 x 0.02 = 0.48 s` of policy time
- mini-batch size is `(num_envs x 24) / 4`

## 11. AMP Code Still Present In The Repo

The codebase still contains AMP support, but it is not used by the current default task configuration.

Dormant AMP settings in the repo:

- AMP observation width: `23`
- AMP frame count: `5`
- AMP motion source: `dataset/roller_push`
- AMP preload transitions: `200000`
- AMP discriminator hidden dims: `(256, 256)`
- AMP reward coefficient: `5.0`
- `use_lerp = False`
- `amp_task_reward_lerp = 0.7` is present in config but not used when `use_lerp = False`

Current discriminator setup:

- input width: `23 x 5 = 115`
- trunk activation: `ReLU`
- trunk layers: `115 -> 256 -> 256`
- output head: linear to `1`
- spectral normalization: enabled on discriminator linear layers

How AMP would apply if re-enabled:

- AMP observations are the 23 controlled joint positions
- the AMP runner forms 5-frame sequences
- AMP reward would only apply where `amp_active_mask` is true
- in the environment, `amp_active_mask = push_phase`

So in the current repo:

- task rewards shape the whole cycle
- the default task does not add imitation reward

## 12. Practical Reading Order

If you want to understand the current system quickly, read these files in order:

1. `src/mjlab_roller/tasks/roller/config/g1/rl_cfg.py`
2. `src/mjlab_roller/tasks/roller/roller_env_cfg.py`
3. `src/mjlab_roller/tasks/roller/config/g1/env_cfgs.py`
4. `src/mjlab_roller/envs/g1_roller_rl_env.py`
5. `src/mjlab_roller/tasks/roller/mdp/rewards.py`
6. `src/mjlab_roller/tasks/roller/mdp/velocity_command.py`
7. `rsl_rl/modules/actor_critic.py`
8. `rsl_rl/runners/on_policy_runner.py`
