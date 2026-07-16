from __future__ import annotations

from typing import TYPE_CHECKING

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
  z_bias = torch.zeros(env.num_envs, 1, device=env.device)
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


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase
