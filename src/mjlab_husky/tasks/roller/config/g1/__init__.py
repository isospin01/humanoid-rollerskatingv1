from mjlab_husky.tasks.registry import register_mjlab_task
from mjlab_husky.tasks.roller.rl import RollerOnPolicyRunner

from .env_cfgs import unitree_g1_roller_env_cfg
from .rl_cfg import unitree_g1_roller_ppo_runner_cfg


register_mjlab_task(
  task_id="Mjlab-Roller-Flat-Unitree-G1",
  env_cfg=unitree_g1_roller_env_cfg(),
  play_env_cfg=unitree_g1_roller_env_cfg(play=True),
  rl_cfg=unitree_g1_roller_ppo_runner_cfg(),
  runner_cls=RollerOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Skater-Flat-Unitree-G1",
  env_cfg=unitree_g1_roller_env_cfg(),
  play_env_cfg=unitree_g1_roller_env_cfg(play=True),
  rl_cfg=unitree_g1_roller_ppo_runner_cfg(),
  runner_cls=RollerOnPolicyRunner,
)
