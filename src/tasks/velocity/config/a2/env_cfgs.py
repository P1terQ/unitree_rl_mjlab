"""Unitree A2 velocity environment configurations."""

from dataclasses import dataclass

from typing import Literal

from src.assets.robots import (
  get_a2_robot_cfg,
)
import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.config import (
  discrete_obstacles,
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
  stepping_stones,
)
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


def make_a2_locoscan_terrain_cfg() -> TerrainGeneratorCfg:
  """Create a terrain mix matching the legacy A2 LocoScan proportions."""
  return TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=25.0,
    border_height=0.01,
    num_rows=10,
    num_cols=20,
    sub_terrains={
      "hf_pyramid_slope": hf_pyramid_slope(proportion=0.15, slope_range=(0.0, 0.7)),
      "random_rough": random_rough(proportion=0.15),
      "pyramid_stairs": pyramid_stairs(
        proportion=0.2,
        step_height_range=(0.01, 0.2),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
      ),
      "pyramid_stairs_inv": pyramid_stairs_inv(
        proportion=0.2,
        step_height_range=(0.01, 0.2),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
      ),
      "discrete_obstacles": discrete_obstacles(proportion=0.2),
      "stepping_stones": stepping_stones(proportion=0.1),
      # Keep the old "wide step" slot explicit; mjlab has no exact preset.
      "wide_step_proxy": hf_pyramid_slope_inv(
        proportion=0.0,
        slope_range=(0.0, 0.7),
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
        cfg.scene.terrain.terrain_generator.border_width = 10.0

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
  cfg.scene.terrain.max_init_terrain_level = 2

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.pattern = LocoScanGridPatternCfg()
      sensor.max_distance = 5.0
      sensor.debug_vis = False
      sensor.viz.show_normals = False

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
      func=mdp.locoscan_height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      delay_min_lag=5,
      delay_max_lag=9,
      delay_hold_prob=0.2,
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
  cfg.rewards["foot_clearance"].weight = 0.2
  cfg.rewards["foot_clearance"].func = mdp.feet_clearance
  cfg.rewards["foot_clearance"].params["target_height"] = 0.07
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].weight = -0.05
  cfg.rewards["soft_landing"].weight = -0.1
  cfg.rewards["stand_still"].weight = -0.2
  cfg.rewards["pose"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["joint_pos_limits"].weight = 0.0
  cfg.rewards["is_terminated"].weight = 0.0

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
    weight=-2.0,
    params={"sensor_name": "feet_ground_contact"},
  )
  cfg.rewards["feet_near_terrain_edge_new"] = RewardTermCfg(
    func=mdp.feet_near_terrain_edge,
    weight=-1.0,
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
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg
