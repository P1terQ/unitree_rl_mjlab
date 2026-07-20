"""Unitree A2 velocity environment configurations."""

from dataclasses import dataclass
import uuid

from typing import Literal

from src.assets.robots import (
  get_a2_robot_cfg,
)
import mujoco
import numpy as np
import scipy.interpolate as interpolate
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
import mjlab.terrains as terrain_gen
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.heightfield_terrains import color_by_height
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
import src.tasks.velocity.mdp as mdp

TerrainType = Literal["rough", "obstacles"]

LOCOSCAN_POINTS_X = (
  -0.5,
  -0.4,
  -0.3,
  -0.2,
  -0.1,
  0.0,
  0.1,
  0.2,
  0.3,
  0.4,
  0.5,
  0.6,
  0.7,
  0.8,
)
LOCOSCAN_POINTS_Y = (
  -0.5,
  -0.4,
  -0.3,
  -0.2,
  -0.1,
  0.0,
  0.1,
  0.2,
  0.3,
  0.4,
  0.5,
)
LOCOSCAN_NUM_SCAN = len(LOCOSCAN_POINTS_X) * len(LOCOSCAN_POINTS_Y)
LOCOSCAN_PROPRIO_DIM = 42
LOCOSCAN_HISTORY_LENGTH = 5


@dataclass
class LocoScanGridPatternCfg(GridPatternCfg):
  """Explicit legacy LocoScan sample points in the base yaw frame."""

  points_x: tuple[float, ...] = LOCOSCAN_POINTS_X
  points_y: tuple[float, ...] = LOCOSCAN_POINTS_Y

  def generate_rays(
    self, mj_model: mujoco.MjModel | None, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del mj_model
    x = torch.tensor(self.points_x, device=device, dtype=torch.float32)
    y = torch.tensor(self.points_y, device=device, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    local_offsets = torch.zeros((grid_x.numel(), 3), device=device, dtype=torch.float32)
    local_offsets[:, 0] = grid_x.reshape(-1)
    local_offsets[:, 1] = grid_y.reshape(-1)
    direction = torch.tensor(self.direction, device=device, dtype=torch.float32)
    direction = direction / direction.norm()
    local_directions = direction.unsqueeze(0).expand(local_offsets.shape[0], 3).clone()
    return local_offsets, local_directions


@dataclass(kw_only=True)
class CenteredHfRandomUniformTerrainCfg(terrain_gen.HfRandomUniformTerrainCfg):
  """Random rough heightfield with the IsaacGym zero-height convention."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty
    body = spec.body("terrain")

    if self.border_width > 0 and self.border_width < self.horizontal_scale:
      raise ValueError(
        f"Border width ({self.border_width}) must be >= horizontal scale "
        f"({self.horizontal_scale})"
      )

    if self.downsampled_scale is None:
      downsampled_scale = self.horizontal_scale
    elif self.downsampled_scale < self.horizontal_scale:
      raise ValueError(
        f"Downsampled scale must be >= horizontal scale: "
        f"{self.downsampled_scale} < {self.horizontal_scale}"
      )
    else:
      downsampled_scale = self.downsampled_scale

    border_pixels = int(self.border_width / self.horizontal_scale)
    width_pixels = int(self.size[0] / self.horizontal_scale)
    length_pixels = int(self.size[1] / self.horizontal_scale)
    noise = np.zeros((width_pixels, length_pixels), dtype=np.int16)

    if border_pixels > 0:
      inner_width_pixels = width_pixels - 2 * border_pixels
      inner_length_pixels = length_pixels - 2 * border_pixels
      terrain_size = (
        inner_width_pixels * self.horizontal_scale,
        inner_length_pixels * self.horizontal_scale,
      )
    else:
      inner_width_pixels = width_pixels
      inner_length_pixels = length_pixels
      terrain_size = self.size

    width_downsampled = int(terrain_size[0] / downsampled_scale)
    length_downsampled = int(terrain_size[1] / downsampled_scale)
    height_min = int(self.noise_range[0] / self.vertical_scale)
    height_max = int(self.noise_range[1] / self.vertical_scale)
    height_step = int(self.noise_step / self.vertical_scale)
    height_range = np.arange(height_min, height_max + height_step, height_step)

    height_field_downsampled = rng.choice(
      height_range, size=(width_downsampled, length_downsampled)
    )
    x = np.linspace(0, terrain_size[0], width_downsampled)
    y = np.linspace(0, terrain_size[1], length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)
    x_upsampled = np.linspace(0, terrain_size[0], inner_width_pixels)
    y_upsampled = np.linspace(0, terrain_size[1], inner_length_pixels)
    z_upsampled = np.rint(func(x_upsampled, y_upsampled)).astype(np.int16)

    if border_pixels > 0:
      noise[
        border_pixels : -border_pixels if border_pixels else width_pixels,
        border_pixels : -border_pixels if border_pixels else length_pixels,
      ] = z_upsampled
    else:
      noise = z_upsampled

    elevation_min = np.min(noise)
    elevation_max = np.max(noise)
    elevation_range = (
      elevation_max - elevation_min if elevation_max != elevation_min else 1
    )
    max_physical_height = elevation_range * self.vertical_scale
    base_thickness = max_physical_height * self.base_thickness_ratio
    normalized_elevation = (noise - elevation_min) / elevation_range

    unique_id = uuid.uuid4().hex
    field = spec.add_hfield(
      name=f"hfield_{unique_id}",
      size=[
        self.size[0] / 2,
        self.size[1] / 2,
        max_physical_height,
        base_thickness,
      ],
      nrow=noise.shape[0],
      ncol=noise.shape[1],
      userdata=normalized_elevation.flatten().astype(np.float32).tolist(),
    )

    hfield_z_offset = elevation_min * self.vertical_scale
    material_name = color_by_height(spec, noise, unique_id, normalized_elevation)
    hfield_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname=field.name,
      pos=[self.size[0] / 2, self.size[1] / 2, hfield_z_offset],
      material=material_name,
    )

    center_x = width_pixels // 2
    center_y = length_pixels // 2
    patch_half_size = max(1, int(1.0 / self.horizontal_scale))
    x0 = max(center_x - patch_half_size, 0)
    x1 = min(center_x + patch_half_size, noise.shape[0])
    y0 = max(center_y - patch_half_size, 0)
    y1 = min(center_y + patch_half_size, noise.shape[1])
    spawn_height = np.max(noise[x0:x1, y0:y1]) * self.vertical_scale
    origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

    geom = TerrainGeometry(geom=hfield_geom, hfield=field)
    return TerrainOutput(origin=origin, geometries=[geom], flat_patches=None)


def make_a2_locoscan_terrain_cfg() -> TerrainGeneratorCfg:
  """Create the full IsaacGym-style curriculum terrain for A2 LocoScan."""
  return TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=0.0,
    border_height=0.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
      "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.075,
        slope_range=(0.0, 0.6),
        platform_width=2.0,
        border_width=0.0,
        horizontal_scale=0.1,
      ),
      "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.075,
        slope_range=(0.0, 0.6),
        platform_width=2.0,
        inverted=True,
        border_width=0.0,
        horizontal_scale=0.1,
      ),
      "random_rough": CenteredHfRandomUniformTerrainCfg(
        proportion=0.15,
        noise_range=(-0.1, 0.1),
        noise_step=0.005,
        downsampled_scale=0.2,
        border_width=0.0,
        horizontal_scale=0.1,
      ),
      "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
        proportion=0.2,
        step_height_range=(0.05, 0.23),
        step_width=0.3,
        platform_width=2.0,
        border_width=0.0,
      ),
      "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
        proportion=0.2,
        step_height_range=(0.05, 0.23),
        step_width=0.3,
        platform_width=2.0,
        border_width=0.0,
      ),
      "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        proportion=0.2,
        obstacle_width_range=(1.0, 2.0),
        obstacle_height_range=(0.05, 0.2),
        num_obstacles=20,
        platform_width=2.0,
        border_width=0.0,
        horizontal_scale=0.1,
      ),
      "wide_step_proxy": terrain_gen.BoxPyramidStairsTerrainCfg(
        proportion=0.1,
        step_height_range=(0.1, 0.3),
        step_width=1.0,
        platform_width=2.0,
        border_width=0.0,
      ),
    },
    add_lights=True,
  )


def unitree_a2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree A2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 256

  cfg.scene.entities = {"robot": get_a2_robot_cfg()}

  # Set raycast sensor frame to A2 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 0.0

  return cfg


def unitree_a2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree A2 flat terrain velocity configuration."""
  cfg = unitree_a2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_a2_locoscan_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree A2 LocoScan terrain-scanning velocity configuration."""
  cfg = unitree_a2_rough_env_cfg(play=play)

  robot = cfg.scene.entities["robot"]
  robot.init_state.pos = (0.0, 0.0, 0.45)
  robot.init_state.joint_pos = {
    "FL_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "FR_thigh_joint": 0.8,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
  }
  for actuator in robot.articulation.actuators:
    actuator.stiffness = 80.0
    actuator.damping = 2.0

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 0.25

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = make_a2_locoscan_terrain_cfg()
  cfg.scene.terrain.max_init_terrain_level = 0
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.nconmax = 256
  cfg.events.pop("push_robot", None)
  if "base_com" in cfg.events:
    cfg.events["base_com"].params["ranges"] = {
      0: (-0.03, 0.03),
      1: (-0.03, 0.03),
      2: (-0.03, 0.03),
    }

  base_ground_cfg = ContactSensorCfg(
    name="base_ground_contact",
    primary=ContactMatch(mode="body", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (base_ground_cfg,)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": base_ground_cfg.name, "force_threshold": 10.0},
  )

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.pattern = LocoScanGridPatternCfg()
      sensor.max_distance = 5.0
      sensor.debug_vis = play
      sensor.viz.show_rays = False
      sensor.viz.show_normals = False
      sensor.viz.hit_sphere_radius = 0.12

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.resampling_time_range = (10.0, 10.0)
  twist_cmd.rel_standing_envs = 0.2
  twist_cmd.heading_command = True
  twist_cmd.heading_control_stiffness = 0.5
  twist_cmd.ranges.lin_vel_x = (0.0, 0.5)
  twist_cmd.ranges.lin_vel_y = (-0.25, 0.25)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  twist_cmd.ranges.heading = (-1.0, 1.0)

  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "proprioception": ObservationTermCfg(
      func=mdp.locoscan_proprioception,
      history_length=LOCOSCAN_HISTORY_LENGTH,
      noise=Unoise(
        n_min=(
          *([-0.2] * 3),
          *([-0.05] * 3),
          *([-0.01] * 12),
          *([-1.5] * 12),
          *([0.0] * 12),
        ),
        n_max=(
          *([0.2] * 3),
          *([0.05] * 3),
          *([0.01] * 12),
          *([1.5] * 12),
          *([0.0] * 12),
        ),
      ),
    ),
    "height_scan": ObservationTermCfg(
      func=mdp.LocoScanNoisyHeightScan,
      params={
        "sensor_name": "terrain_scan",
        "scan_shape": (len(LOCOSCAN_POINTS_X), len(LOCOSCAN_POINTS_Y)),
        "points_x": LOCOSCAN_POINTS_X,
        "points_y": LOCOSCAN_POINTS_Y,
        "per_step_xy_noise_std": 0.02,
        "per_step_z_noise_std": 0.025,
        "per_env_resampling_s": 7.0,
        "per_env_noise_prob": 0.3,
        "per_env_xy_noise_std": 0.05,
        "per_env_z_noise_std": 0.05,
        "scan_map_tilt_prob": 0.3,
        "scan_map_tilt_max_rad": 0.07,
        "delay_prob_start": 0.2,
        "delay_prob_end": 0.8,
        "delay_min_steps": 5,
        "delay_max_steps": 9,
        "noise_curriculum": True,
        "noise_curriculum_start_level": 2,
        "noise_curriculum_end_level": 6,
        "enable_noise": not play,
      },
    ),
  }
  critic_terms = {
    "estimator_target": ObservationTermCfg(
      func=mdp.locoscan_estimator_target,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "proprioception": ObservationTermCfg(
      func=mdp.locoscan_proprioception,
      history_length=LOCOSCAN_HISTORY_LENGTH,
    ),
    "height_scan": ObservationTermCfg(
      func=mdp.locoscan_height_scan,
      params={"sensor_name": "terrain_scan"},
    ),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  site_names = ("FR", "FL", "RR", "RL")
  joint_names = (".*",)
  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["track_angular_velocity"].weight = 1.0
  cfg.rewards["body_orientation_l2"].weight = -0.5
  cfg.rewards["body_ang_vel"].weight = -0.15
  cfg.rewards["joint_acc_l2"].weight = -2.5e-7
  cfg.rewards["action_rate_l2"].weight = -0.02
  cfg.rewards["foot_gait"].weight = 1.0
  cfg.rewards["foot_clearance"].weight = -1.0
  cfg.rewards["foot_clearance"].func = mdp.feet_clearance
  cfg.rewards["foot_clearance"].params["target_height"] = 0.10
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].weight = -0.05
  cfg.rewards["soft_landing"].weight = 0.0
  cfg.rewards["stand_still"].weight = -0.2
  cfg.rewards["pose"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["joint_pos_limits"].weight = 0.0
  cfg.rewards["is_terminated"].weight = -200.0

  cfg.rewards["base_height_exp"] = RewardTermCfg(
    func=mdp.base_height_exp,
    weight=0.1,
    params={"target_height": 0.42, "std": 0.005},
  )
  cfg.rewards["base_height_l2"] = RewardTermCfg(
    func=mdp.base_height_l2,
    weight=-1.0,
    params={"target_height": 0.42},
  )
  cfg.rewards["no_backward_on_forward_cmd"] = RewardTermCfg(
    func=mdp.no_backward_on_forward_cmd,
    weight=-1.5,
    params={"command_name": "twist", "command_threshold": 0.1},
  )
  cfg.rewards["hip_pos"] = RewardTermCfg(
    func=mdp.joint_deviation_l2,
    weight=-0.15,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=(r".*(FR|FL|RR|RL)_hip_joint.*",)
      )
    },
  )
  cfg.rewards["default_dof_pos"] = RewardTermCfg(
    func=mdp.joint_deviation_l2,
    weight=-0.05,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)},
  )
  cfg.rewards["feet_stumble_swing"] = RewardTermCfg(
    func=mdp.feet_stumble_swing,
    weight=0.0,
    params={"sensor_name": "feet_ground_contact"},
  )
  cfg.rewards["feet_near_terrain_edge_new"] = RewardTermCfg(
    func=mdp.feet_near_terrain_edge,
    weight=0.0,
    params={
      "sensor_name": "terrain_scan",
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      "scan_shape": (len(LOCOSCAN_POINTS_X), len(LOCOSCAN_POINTS_Y)),
      "edge_threshold": 0.08,
      "foot_radius": 0.08,
    },
  )
  cfg.rewards["foot_landing_vel"] = RewardTermCfg(
    func=mdp.foot_landing_vel,
    weight=-0.1,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (0.0, 0.5),
        "lin_vel_y": (-0.25, 0.25),
        "ang_vel_z": (-1.0, 1.0),
      },
    ]

  if play:
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 0.0

  return cfg
