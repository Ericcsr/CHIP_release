import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MotionOnPolicyRunner(OnPolicyRunner):
    """On-policy runner that additionally exports an ONNX policy next to every saved checkpoint."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.disable_logs:
            return
        policy_path = os.path.dirname(path)
        filename = os.path.basename(policy_path) + ".onnx"
        export_motion_policy_as_onnx(
            self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
        )
        run_name = os.path.basename(policy_path)
        if self.logger_type == "wandb":
            import wandb

            run_name = wandb.run.name
        attach_onnx_metadata(self.env.unwrapped, run_name, path=policy_path, filename=filename)
        if self.logger_type == "wandb":
            wandb.save(os.path.join(policy_path, filename), base_path=os.path.dirname(policy_path))
