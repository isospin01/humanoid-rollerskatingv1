"""RL configuration for the Unitree G1 roller task."""

from mjlab.rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from mjlab_husky.rl import RslRlAMPOnPolicyRunnerCfg


def unitree_g1_roller_ppo_runner_cfg() -> RslRlAMPOnPolicyRunnerCfg:
  return RslRlAMPOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      class_name="AMP_PPO",
    ),
    experiment_name="g1_roller",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=50_000,
  )
