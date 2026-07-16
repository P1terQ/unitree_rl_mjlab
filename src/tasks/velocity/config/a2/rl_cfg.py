"""RL configuration for Unitree A2 velocity task."""

from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class LocoScanActorCfg(RslRlModelCfg):
  class_name: str = "src.tasks.velocity.rl.locoscan:ActorCriticLocoScan"
  hidden_dims: tuple[int, ...] = (512, 256, 128)
  estimator_hidden_dims: tuple[int, ...] = (256, 128)
  scan_hidden_dims: tuple[int, ...] = (128, 64)
  scan_input_dims: int = 154
  scan_latent_dims: int = 32
  estimator_output_dims: int = 8
  activation: str = "elu"
  obs_normalization: bool = True
  distribution_cfg: dict = field(
    default_factory=lambda: {
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    }
  )


@dataclass
class LocoScanPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "src.tasks.velocity.rl.locoscan:PPOLocoScan"
  learning_rate: float = 5.0e-4
  sym_loss: bool = True
  sym_loss_coef: float = 0.5
  estimator_loss_coef: float = 1.0
  frame_stack: int = 5
  num_scan: int = 154
  policy_obs_permutation: list[float] = field(
    default_factory=lambda: [
      -0.0001, 1, -2,
      -3, 4, -5,
      -9, 10, 11, -6, 7, 8, -15, 16, 17, -12, 13, 14,
      -21, 22, 23, -18, 19, 20, -27, 28, 29, -24, 25, 26,
      -33, 34, 35, -30, 31, 32, -39, 40, 41, -36, 37, 38,
    ]
  )
  act_permutation: list[float] = field(
    default_factory=lambda: [-3, 4, 5, -0.0001, 1, 2, -9, 10, 11, -6, 7, 8]
  )


def unitree_a2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree A2 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="a2_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def unitree_a2_locoscan_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree A2 LocoScan task."""
  return RslRlOnPolicyRunnerCfg(
    class_name="src.tasks.velocity.rl.locoscan:LocoScanRunner",
    actor=LocoScanActorCfg(),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=LocoScanPpoAlgorithmCfg(),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    experiment_name="a2_locoscan",
    run_name="a2_locoscan_mjlab",
    save_interval=1000,
    num_steps_per_env=24,
    max_iterations=80000,
  )
