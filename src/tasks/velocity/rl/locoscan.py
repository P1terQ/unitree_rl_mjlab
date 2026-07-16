from __future__ import annotations

import copy
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config
from rsl_rl.modules import EmpiricalNormalization, MLP, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import (
  compile_model,
  resolve_callable,
  resolve_nn_activation,
  resolve_obs_groups,
  resolve_optimizer,
)
from tensordict import TensorDict

from mjlab.rl.runner import MjlabOnPolicyRunner


class ActorCriticLocoScan(MLPModel):
  """LocoScan actor with proprioceptive estimator and height-scan encoder."""

  is_recurrent = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] | None = None,
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128),
    estimator_hidden_dims: tuple[int, ...] = (256, 128),
    scan_hidden_dims: tuple[int, ...] = (128, 64),
    scan_input_dims: int = 154,
    scan_latent_dims: int = 32,
    estimator_output_dims: int = 8,
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    cnn_cfg: dict | None = None,
    rnn_type: str | None = None,
    rnn_hidden_dim: int | None = None,
    rnn_num_layers: int | None = None,
  ) -> None:
    nn.Module.__init__(self)
    del cnn_cfg, rnn_type, rnn_hidden_dim, rnn_num_layers
    if hidden_dims is not None:
      actor_hidden_dims = tuple(hidden_dims)
    self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
    self.scan_input_dims = scan_input_dims
    self.scan_latent_dims = scan_latent_dims
    self.estimator_output_dims = estimator_output_dims
    self.proprio_dim = self.obs_dim - 3 - self.scan_input_dims
    if self.proprio_dim <= 0:
      raise ValueError(
        "LocoScan actor expects obs layout command + proprio_history + scan, "
        f"got obs_dim={self.obs_dim}, scan_input_dims={self.scan_input_dims}."
      )

    self.obs_normalization = obs_normalization
    self.obs_normalizer = (
      EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
    )

    if distribution_cfg is not None:
      dist_cfg = dict(distribution_cfg)
      dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore[assignment]
      self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
      actor_output_dim = self.distribution.input_dim
    else:
      self.distribution = None
      actor_output_dim = output_dim

    self.estimator_module = MLP(
      self.proprio_dim,
      self.estimator_output_dims,
      estimator_hidden_dims,
      activation,
    )
    self.scan_encoder = MLP(
      self.scan_input_dims,
      self.scan_latent_dims,
      scan_hidden_dims,
      activation,
    )
    self.mlp = MLP(
      3 + self.proprio_dim + self.estimator_output_dims + self.scan_latent_dims,
      actor_output_dim,
      actor_hidden_dims,
      activation,
    )
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    raw_obs = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
    norm_obs = self.obs_normalizer(raw_obs)
    command = norm_obs[:, :3]
    proprio = norm_obs[:, 3:-self.scan_input_dims]
    scan = norm_obs[:, -self.scan_input_dims:]
    estimated = self.estimator_module(proprio)
    scan_latent = self.scan_encoder(scan)
    return torch.cat((command, proprio, estimated, scan_latent), dim=-1)

  def get_estimated_val(self, obs: TensorDict) -> torch.Tensor:
    raw_obs = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
    norm_obs = self.obs_normalizer(raw_obs)
    proprio = norm_obs[:, 3:-self.scan_input_dims]
    return self.estimator_module(proprio)

  def as_jit(self) -> nn.Module:
    return _TorchLocoScanModel(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _OnnxLocoScanModel(self, verbose)


class _TorchLocoScanModel(nn.Module):
  def __init__(self, model: ActorCriticLocoScan) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.estimator_module = copy.deepcopy(model.estimator_module)
    self.scan_encoder = copy.deepcopy(model.scan_encoder)
    self.mlp = copy.deepcopy(model.mlp)
    self.scan_input_dims = model.scan_input_dims
    self.deterministic_output = (
      model.distribution.as_deterministic_output_module()
      if model.distribution is not None
      else nn.Identity()
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.obs_normalizer(x)
    command = x[:, :3]
    proprio = x[:, 3:-self.scan_input_dims]
    scan = x[:, -self.scan_input_dims:]
    estimated = self.estimator_module(proprio)
    scan_latent = self.scan_encoder(scan)
    out = self.mlp(torch.cat((command, proprio, estimated, scan_latent), dim=-1))
    return self.deterministic_output(out)

  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxLocoScanModel(_TorchLocoScanModel):
  is_recurrent = False

  def __init__(self, model: ActorCriticLocoScan, verbose: bool) -> None:
    super().__init__(model)
    self.verbose = verbose
    self.input_size = model.obs_dim

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


class PPOLocoScan(PPO):
  """PPO with legacy LocoScan estimator and mirror losses."""

  def __init__(
    self,
    actor: ActorCriticLocoScan,
    critic: MLPModel,
    storage: RolloutStorage,
    *args,
    estimator_loss_coef: float = 1.0,
    sym_loss: bool = False,
    policy_obs_permutation: list[float] | None = None,
    act_permutation: list[float] | None = None,
    frame_stack: int = 5,
    sym_loss_coef: float = 1.0,
    num_scan: int = 154,
    optimizer: str = "adam",
    learning_rate: float = 1.0e-3,
    **kwargs,
  ) -> None:
    super().__init__(
      actor,
      critic,
      storage,
      *args,
      optimizer=optimizer,
      learning_rate=learning_rate,
      **kwargs,
    )
    self.estimator_loss_coef = estimator_loss_coef
    self.estimator_optimizer = resolve_optimizer(optimizer)(
      self._raw_actor.parameters(), lr=learning_rate
    )
    self.sym_loss = sym_loss
    self.sym_loss_coef = sym_loss_coef
    self.num_scan = num_scan

    self.policy_obs_perm_mat: torch.Tensor | None = None
    self.act_perm_mat: torch.Tensor | None = None
    if self.sym_loss:
      if policy_obs_permutation is None or act_permutation is None:
        raise ValueError("LocoScan symmetry requires observation and action permutations.")
      policy_perm_stack: list[float] = []
      for frame in range(frame_stack):
        offset = frame * len(policy_obs_permutation)
        for perm in policy_obs_permutation:
          policy_perm_stack.append(float(torch.sign(torch.tensor(perm))) * (abs(perm) + offset))
      self.policy_obs_perm_mat = self._build_permutation_matrix(policy_perm_stack, self.device)
      self.act_perm_mat = self._build_permutation_matrix(act_permutation, self.device)

  @staticmethod
  def _build_permutation_matrix(permutation: list[float], device: str) -> torch.Tensor:
    mat = torch.zeros((len(permutation), len(permutation)), device=device)
    for i, perm in enumerate(permutation):
      sign = 1.0 if perm >= 0 else -1.0
      mat[int(abs(perm)), i] = sign
    return mat

  def update(self) -> dict[str, float]:
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_estimator_loss = 0.0
    mean_symmetry_loss = 0.0

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for batch in generator:
      original_batch_size = batch.observations.batch_size[0]
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
          )

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(
        batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
      )
      distribution_params = tuple(
        p[:original_batch_size] for p in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params, distribution_params
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for optimizer in (self.optimizer, self.estimator_optimizer):
            for param_group in optimizer.param_groups:
              param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      action_mean = self.actor.output_mean[:original_batch_size].clone()
      symmetry_loss = self._compute_symmetry_loss(
        batch.observations[:original_batch_size], action_mean
      )
      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
        + self.sym_loss_coef * symmetry_loss
      )

      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      estimator_loss = F.mse_loss(
        self.actor.get_estimated_val(batch.observations),
        batch.observations["critic"][:, : self.actor.estimator_output_dims].detach(),
      )
      self.estimator_optimizer.zero_grad()
      (self.estimator_loss_coef * estimator_loss).backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      self.estimator_optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_estimator_loss += estimator_loss.item()
      mean_symmetry_loss += symmetry_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    return {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "estimator": mean_estimator_loss / num_updates,
      "symmetry": mean_symmetry_loss / num_updates,
    }

  def _compute_symmetry_loss(
    self, obs: TensorDict, action_mean: torch.Tensor
  ) -> torch.Tensor:
    if not self.sym_loss:
      return torch.zeros((), device=self.device)
    assert self.policy_obs_perm_mat is not None and self.act_perm_mat is not None
    policy_obs = obs["actor"]
    command = policy_obs[:, :3]
    proprio = policy_obs[:, 3:-self.num_scan]
    scan = policy_obs[:, -self.num_scan:]
    mirror_command = torch.stack(
      (command[:, 0], -command[:, 1], -command[:, 2]), dim=1
    )
    mirror_proprio = proprio @ self.policy_obs_perm_mat
    mirror_obs = obs.clone(False)
    mirror_obs["actor"] = torch.cat((mirror_command, mirror_proprio, scan), dim=-1)
    mirrored_actions = self.actor(mirror_obs)
    expected_actions = action_mean @ self.act_perm_mat
    return F.mse_loss(mirrored_actions, expected_actions.detach())

  def compile(self, mode: str | None = None) -> None:
    self.actor = compile_model(self._raw_actor, mode)  # type: ignore[assignment]
    self.critic = compile_model(self._raw_critic, mode)  # type: ignore[assignment]

  def reduce_parameters(self) -> None:
    all_params = list(chain(self.actor.parameters(), self.critic.parameters()))
    grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
    if not grads:
      return
    all_grads = torch.cat(grads)
    torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
    all_grads /= self.gpu_world_size
    offset = 0
    for param in all_params:
      if param.grad is None:
        continue
      numel = param.numel()
      param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
      offset += numel

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> PPOLocoScan:
    alg_class: type[PPOLocoScan] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[assignment]
    actor_class: type[ActorCriticLocoScan] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore[assignment]
    critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore[assignment]

    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"]["symmetry_cfg"] = None
    cfg["algorithm"].pop("share_cnn_encoders", None)

    actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
    print(f"Actor Model: {actor}")
    critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
    print(f"Critic Model: {critic}")

    storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
    alg = alg_class(
      actor,
      critic,
      storage,
      device=device,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg["multi_gpu"],
    )
    alg.compile(cfg.get("torch_compile_mode"))
    return alg


class LocoScanRunner(MjlabOnPolicyRunner):
  """Runner marker for LocoScan tasks."""
