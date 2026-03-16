import os

import wandb
from mjlab.rl import RslRlVecEnvWrapper

from rsl_rl.runners import AMPOnPolicyRunner

from mjlab_husky.tasks.roller.rl.exporter import (
  attach_onnx_metadata,
  export_roller_policy_as_onnx,
)


class RollerOnPolicyRunner(AMPOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    if self.logger_type in ["wandb"]:
      policy_path = path.split("model")[0]
      filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
      normalizer = self.alg.policy.actor_obs_normalizer if self.alg.policy.actor_obs_normalization else None
      export_roller_policy_as_onnx(
        self.alg.policy,
        normalizer=normalizer,
        path=policy_path,
        filename=filename,
      )
      attach_onnx_metadata(
        self.env.unwrapped,
        wandb.run.name,  # type: ignore
        path=policy_path,
        filename=filename,
      )
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
