from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.envs import mdp as envs_mdp

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def locoscan_proprioception(env: ManagerBasedRlEnv) -> torch.Tensor:
  """42-D LocoScan proprioception vector used by the legacy policy layout."""
  return torch.cat(
    (
      envs_mdp.builtin_sensor(env, "robot/imu_ang_vel"),
      envs_mdp.projected_gravity(env),
      envs_mdp.joint_pos_rel(env),
      envs_mdp.joint_vel_rel(env),
      envs_mdp.last_action(env),
    ),
    dim=-1,
  )


def locoscan_estimator_target(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Privileged target for the LocoScan estimator: velocity, contacts, z bias."""
  base_lin_vel = envs_mdp.builtin_sensor(env, "robot/imu_lin_vel")
  contact = foot_contact(env, sensor_name)
  z_bias = getattr(env, "_locoscan_scan_z_bias", None)
  if z_bias is None:
    z_bias = torch.zeros(env.num_envs, 1, device=env.device)
  z_bias = z_bias / 0.05
  return torch.cat((base_lin_vel, contact, z_bias), dim=-1)


def locoscan_height_scan(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  base_offset: float = 0.3,
  scale: float = 5.0,
) -> torch.Tensor:
  """Legacy LocoScan-normalized terrain heights."""
  heights = -envs_mdp.height_scan(env, sensor_name=sensor_name) + base_offset
  return torch.clip(heights, -1.0, 1.0) * scale


class LocoScanNoisyHeightScan:
  """Stateful LocoScan scan corruption close to the legacy IsaacGym setup."""

  def __init__(self, cfg: Any, env: ManagerBasedRlEnv):
    params = cfg.params
    self.sensor_name = params["sensor_name"]
    self.base_offset = float(params.get("base_offset", 0.3))
    self.scale = float(params.get("scale", 5.0))
    self.enable_noise = bool(params.get("enable_noise", True))
    self.scan_shape = tuple(params["scan_shape"])
    self.num_scan = int(self.scan_shape[0] * self.scan_shape[1])
    self.history_length = int(params.get("delay_history_length", 10))

    self.per_step_xy_noise_std = float(params.get("per_step_xy_noise_std", 0.02))
    self.per_step_z_noise_std = float(params.get("per_step_z_noise_std", 0.025))
    self.per_env_resampling_s = float(params.get("per_env_resampling_s", 7.0))
    self.per_env_noise_prob = float(params.get("per_env_noise_prob", 0.3))
    self.per_env_xy_noise_std = float(params.get("per_env_xy_noise_std", 0.05))
    self.per_env_z_noise_std = float(params.get("per_env_z_noise_std", 0.05))
    self.scan_map_tilt_prob = float(params.get("scan_map_tilt_prob", 0.3))
    self.scan_map_tilt_max_rad = float(params.get("scan_map_tilt_max_rad", 0.07))
    self.delay_prob_start = float(params.get("delay_prob_start", 0.2))
    self.delay_prob_end = float(params.get("delay_prob_end", 0.8))
    self.delay_min_steps = int(params.get("delay_min_steps", 5))
    self.delay_max_steps = int(params.get("delay_max_steps", 9))
    self.noise_curriculum = bool(params.get("noise_curriculum", True))
    self.noise_curriculum_start_level = float(
      params.get("noise_curriculum_start_level", 2)
    )
    self.noise_curriculum_end_level = float(params.get("noise_curriculum_end_level", 6))

    points_x = torch.tensor(params["points_x"], device=env.device, dtype=torch.float32)
    points_y = torch.tensor(params["points_y"], device=env.device, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(points_x, points_y, indexing="ij")
    self._scan_x = grid_x.reshape(1, self.num_scan)
    self._scan_y = grid_y.reshape(1, self.num_scan)
    self._dx = float(torch.mean(points_x[1:] - points_x[:-1]).item())
    self._dy = float(torch.mean(points_y[1:] - points_y[:-1]).item())

    self._env = env
    self._env_xyz_noise = torch.zeros(env.num_envs, 3, device=env.device)
    self._map_tilt = torch.zeros(env.num_envs, 2, device=env.device)
    self._history = torch.zeros(
      env.num_envs, self.history_length, self.num_scan, device=env.device
    )
    self._last_resample_step = -1
    self._set_env_debug_state(0.0)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(self._env.num_envs, device=self._env.device)
    self._history[env_ids] = 0.0
    self._env_xyz_noise[env_ids] = 0.0
    self._map_tilt[env_ids] = 0.0
    self._set_env_debug_state(self._noise_alpha())

  def __call__(self, env: ManagerBasedRlEnv, **_: Any) -> torch.Tensor:
    clean_scan = -envs_mdp.height_scan(env, sensor_name=self.sensor_name) + self.base_offset
    if not self.enable_noise:
      noisy_scan = clean_scan
      alpha = 0.0
    else:
      alpha = self._noise_alpha()
      self._maybe_resample_env_noise(alpha)
      noisy_scan = self._apply_scan_noise(clean_scan, alpha)
    self._set_env_debug_state(alpha)

    normalized_scan = torch.clip(noisy_scan, -1.0, 1.0) * self.scale
    if not self.enable_noise:
      return normalized_scan
    return self._apply_delay(normalized_scan, alpha)

  def _noise_alpha(self) -> float:
    if not self.enable_noise:
      return 0.0
    if not self.noise_curriculum:
      return 1.0
    terrain = self._env.scene.terrain
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if terrain_levels is None:
      return 0.0
    denom = max(self.noise_curriculum_end_level - self.noise_curriculum_start_level, 1.0e-6)
    alpha = (torch.mean(terrain_levels.float()).item() - self.noise_curriculum_start_level) / denom
    return min(max(alpha, 0.0), 1.0)

  def _maybe_resample_env_noise(self, alpha: float) -> None:
    period_steps = max(1, int(round(self.per_env_resampling_s / self._env.step_dt)))
    step = int(self._env.common_step_counter)
    if step == self._last_resample_step or step % period_steps != 0:
      return
    self._last_resample_step = step

    if torch.rand((), device=self._env.device).item() < self.per_env_noise_prob * alpha:
      xy_std = self.per_env_xy_noise_std * alpha
      z_std = self.per_env_z_noise_std * alpha
    else:
      xy_std = 0.0
      z_std = 0.0
    self._env_xyz_noise[:, :2] = torch.randn_like(self._env_xyz_noise[:, :2]) * xy_std
    self._env_xyz_noise[:, 2] = torch.randn_like(self._env_xyz_noise[:, 2]) * z_std

    self._map_tilt.zero_()
    tilt_prob = self.scan_map_tilt_prob * alpha
    tilt_max = self.scan_map_tilt_max_rad * alpha
    if tilt_prob > 0.0 and tilt_max > 0.0:
      tilt_mask = torch.rand(self._env.num_envs, device=self._env.device) < tilt_prob
      sampled_tilt = (
        2.0 * torch.rand(self._env.num_envs, 2, device=self._env.device) - 1.0
      ) * tilt_max
      self._map_tilt[tilt_mask] = sampled_tilt[tilt_mask]

  def _apply_scan_noise(self, clean_scan: torch.Tensor, alpha: float) -> torch.Tensor:
    noisy_scan = clean_scan.clone()
    if self.per_step_xy_noise_std > 0.0:
      step_xy_noise = torch.randn(
        self._env.num_envs, self.num_scan, 2, device=self._env.device
      ) * self.per_step_xy_noise_std
      noisy_scan += self._xy_to_height_delta(
        clean_scan, step_xy_noise[..., 0], step_xy_noise[..., 1]
      )
    if torch.any(self._env_xyz_noise[:, :2] != 0.0):
      noisy_scan += self._xy_to_height_delta(
        clean_scan,
        self._env_xyz_noise[:, 0:1].expand(-1, self.num_scan),
        self._env_xyz_noise[:, 1:2].expand(-1, self.num_scan),
      )
    noisy_scan += self._env_xyz_noise[:, 2:3]

    if self.per_step_z_noise_std > 0.0:
      noisy_scan += torch.randn_like(noisy_scan) * self.per_step_z_noise_std
    if torch.any(self._map_tilt != 0.0):
      tilt_z = self._scan_x * self._map_tilt[:, 1:2] + self._scan_y * self._map_tilt[:, 0:1]
      noisy_scan += tilt_z
    del alpha
    return noisy_scan

  def _xy_to_height_delta(
    self, heights: torch.Tensor, x_noise: torch.Tensor, y_noise: torch.Tensor
  ) -> torch.Tensor:
    height_grid = heights.reshape(self._env.num_envs, *self.scan_shape)
    grad_x = torch.zeros_like(height_grid)
    grad_y = torch.zeros_like(height_grid)
    grad_x[:, 1:-1, :] = (height_grid[:, 2:, :] - height_grid[:, :-2, :]) / (2.0 * self._dx)
    grad_x[:, 0, :] = (height_grid[:, 1, :] - height_grid[:, 0, :]) / self._dx
    grad_x[:, -1, :] = (height_grid[:, -1, :] - height_grid[:, -2, :]) / self._dx
    grad_y[:, :, 1:-1] = (height_grid[:, :, 2:] - height_grid[:, :, :-2]) / (2.0 * self._dy)
    grad_y[:, :, 0] = (height_grid[:, :, 1] - height_grid[:, :, 0]) / self._dy
    grad_y[:, :, -1] = (height_grid[:, :, -1] - height_grid[:, :, -2]) / self._dy
    return grad_x.reshape(self._env.num_envs, self.num_scan) * x_noise + grad_y.reshape(
      self._env.num_envs, self.num_scan
    ) * y_noise

  def _apply_delay(self, scan: torch.Tensor, alpha: float) -> torch.Tensor:
    reset_mask = self._env.episode_length_buf <= 1
    if torch.any(reset_mask):
      self._history[reset_mask] = scan[reset_mask].unsqueeze(1).expand(
        -1, self.history_length, -1
      )
    if torch.any(~reset_mask):
      self._history[~reset_mask, :-1] = self._history[~reset_mask, 1:].clone()
      self._history[~reset_mask, -1] = scan[~reset_mask]

    delay_prob = self.delay_prob_start + alpha * (self.delay_prob_end - self.delay_prob_start)
    delay_prob = min(max(delay_prob, 0.0), 1.0)
    use_delay = torch.rand(self._env.num_envs, device=self._env.device) < delay_prob
    high = min(self.delay_max_steps, self.history_length)
    low = min(self.delay_min_steps, high - 1)
    delayed_steps = torch.randint(low, high, (self._env.num_envs,), device=self._env.device)
    current_steps = torch.ones(self._env.num_envs, dtype=torch.long, device=self._env.device)
    delay_steps = torch.where(use_delay, delayed_steps, current_steps)
    return self._history[torch.arange(self._env.num_envs, device=self._env.device), -delay_steps]

  def _set_env_debug_state(self, alpha: float) -> None:
    self._env._locoscan_scan_z_bias = self._env_xyz_noise[:, 2:3]
    self._env._locoscan_scan_noise_alpha = alpha


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase
