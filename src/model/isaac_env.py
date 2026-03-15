import torch
import math
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import RayCasterCfg, patterns, ImuCfg
from isaaclab.managers import SceneEntityCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg, EventTermCfg, ActionTerm, ActionTermCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfRandomUniformTerrainCfg, HfDiscreteObstaclesTerrainCfg, FlatPatchSamplingCfg
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
import os
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

# --- 1. Scene Configuration ---
@configclass
class LynxmotionSceneCfg(InteractiveSceneCfg):
    """Configuration for the Lynxmotion A4WD1 scene."""
    
    # Ground Plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
            size=(20.0, 20.0),
            border_width=1.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.001,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    proportion=1.0,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(0.5, 1.0),
                    obstacle_height_range=(0.4, 0.4),
                    num_obstacles=40,
                    platform_width=2.0,
                    flat_patch_sampling={
                        "init_pos": FlatPatchSamplingCfg(
                            num_patches=100,
                            patch_radius=[0.3, 0.4, 0.5],
                            max_height_diff=0.1,
                            x_range=(-4.5, 4.5),
                            y_range=(-4.5, 4.5),
                        ),
                        "target_pos": FlatPatchSamplingCfg(
                            num_patches=100,
                            patch_radius=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            max_height_diff=0.1,
                            x_range=(-4.5, 4.5),
                            y_range=(-4.5, 4.5),
                        )
                    }
                ),
            },
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Natural/Dirt.mdl",
            project_uvw=True,
        ),
    )

    # Mars-like lighting
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1500.0,
            color=(1.0, 0.85, 0.7),  # warm Mars-like sunlight
        ),
    )

    # THE ROBOT: 4-Wheel Skid Steer Configuration
    # Note: If you have your own A4WD1 USD, change 'usd_path' and 'joint_names' accordingly.
    # We use the Husky here as a placeholder because it has the same 4-wheel kinematics.
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UrdfFileCfg(
            asset_path=f"{os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets', 'lynx.urdf'))}", 
            activate_contact_sensors=False,
            make_instanceable=False,
            fix_base=False,
            joint_drive=UrdfFileCfg.JointDriveCfg(
                gains=UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0)
            )
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1), # Ground level spawn to prevent dropping frames
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["joint_fl", "joint_fr", "joint_rl", "joint_rr"],
                effort_limit_sim=20.0,
                velocity_limit_sim=15.0,
                stiffness=0.0,
                damping=1.0,
            ),
        },
    )

    # Target (Red Cylinder)
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CylinderCfg(
            radius=0.15,
            height=0.4,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
    )

    # Obstacles are now procedurally baked into the static terrain mesh via HfDiscreteObstaclesTerrainCfg

    # Lidar Sensor (24 rays, 360 degrees)
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        # Centered on Y-axis for symmetrical perception
        offset=RayCasterCfg.OffsetCfg(pos=(0.10, 0.0, 0.18)), 
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0, 360),
            horizontal_res=15.0, # 360 / 15 = 24 rays
        ),
        debug_vis=True, 
        # Ensure this covers the terrain and baked obstacles
        mesh_prim_paths=["/World/ground"], 
        max_distance=10.0, # Match your observation/reward scaling
    )

    # IMU Sensor
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        # Adding a small offset if the Arduino isn't exactly at the center of mass
        offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    )

# --- 2. Observation Functions ---
def obs_current_pos(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    imu = env.scene[sensor_cfg.name]
    # Normalise to [-1, 1] so all obs are in the same range for the NN.
    # Max forward speed ~1.5 m/s (wheel vel_limit=15 rad/s * radius~0.1m).
    # Max yaw rate ~3.0 rad/s (empirically safe upper bound for this rover).
    lin_vel_x = torch.clamp(imu.data.lin_vel_b[:, 0] / 1.5, -1.0, 1.0)
    ang_vel_z = torch.clamp(imu.data.ang_vel_b[:, 2] / 3.0, -1.0, 1.0)
    return torch.stack([lin_vel_x, ang_vel_z], dim=1)

def obs_target_pos(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    target = env.scene[sensor_cfg.name]
    robot = env.scene["robot"]
    
    # 1. Get relative vector (Target - Robot) in World Frame
    target_vec_w = target.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    
    # 2. Calculate Distance (r)
    r = torch.norm(target_vec_w, dim=1)
    
    # 3. Calculate Angle to target in World Frame
    target_yaw_w = torch.atan2(target_vec_w[:, 1], target_vec_w[:, 0])
    
    # 4. Get Robot's Current Yaw in World Frame from Quaternion
    quat = robot.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    robot_yaw_w = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    
    # 5. Calculate Relative Heading (theta)
    # This is the angle the robot needs to turn to face the target
    theta = target_yaw_w - robot_yaw_w
    
    # Wrap angle to [-pi, pi] to keep the observation continuous
    theta = torch.atan2(torch.sin(theta), torch.cos(theta))
    
    # Normalize r (assuming 10m max) and theta (divided by pi to get [-1, 1])
    r_norm = torch.clamp(r / 10.0, 0.0, 1.0)
    theta_norm = theta / math.pi
    
    return torch.stack([r_norm, theta_norm], dim=1)

def obs_lidar(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    lidar = env.scene[sensor_cfg.name]
    origins = lidar.data.pos_w 
    hits = lidar.data.ray_hits_w 
    
    # Calculate Euclidean distance [num_envs, num_rays]
    dists = torch.norm(hits - origins.unsqueeze(1), dim=-1)
    
    # Clip to 10.0m max to ensure the observation stays within [0, 1]
    # This also handles 'infinite' rays that don't hit anything.
    dists_clipped = torch.clamp(dists, 0.0, 10.0)
    
    # Linear scale: 0m -> 0.0, 10m -> 1.0
    normalized_dists = dists_clipped / 10.0
    
    return normalized_dists

# --- 3. Rewards ---
def rew_distance(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    
    target_dist = torch.norm(target.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    
    max_dist = 10*math.sqrt(2)  # Max distance in a 10x10m arena (diagonal)
    norm_dist = torch.clamp(target_dist / max_dist, 0.0, 1.0)
    
    # +1.0 when within 0.3m. Note: this triggers for the final step of the episode.
    bonus = (target_dist < 0.3).float()

    return bonus - norm_dist

def rew_collision(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    lidar = env.scene["lidar"]
    origins = lidar.data.pos_w
    hits = lidar.data.ray_hits_w

    dists = torch.norm(hits - origins.unsqueeze(1), dim=-1)
    min_dist = torch.min(dists, dim=1)[0]

    max_sensor_range = 10.0
    norm_obs_dist = torch.clamp(min_dist / max_sensor_range, 0.0, 1.0)
    
    # Penalty 'spike' for breaching the 0.3m safety zone
    penalty = (min_dist < 0.3).float()

    # Range: 0.0 (Safe) to -2.0 (Collision)
    return -(1.0 - norm_obs_dist) - penalty

def terminate_success(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    target_dist = torch.norm(target.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    return target_dist < 0.3

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        pos = ObservationTermCfg(func=obs_current_pos, params={"sensor_cfg": SceneEntityCfg("imu")})
        target = ObservationTermCfg(func=obs_target_pos, params={"sensor_cfg": SceneEntityCfg("target")})
        lidar = ObservationTermCfg(func=obs_lidar, params={"sensor_cfg": SceneEntityCfg("lidar")})

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


def reset_target_state_from_terrain(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target"),
):
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain

    valid_positions = terrain.flat_patches.get("target_pos")
    if valid_positions is None:
        raise ValueError("The event term requires valid flat patches under 'target_pos'.")

    ids = torch.randint(0, valid_positions.shape[2], size=(len(env_ids),), device=env.device)
    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
    positions += asset.data.default_root_state[env_ids, :3]

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = asset.data.default_root_state[env_ids, 7:13] + rand_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


# --- Custom Skid-Steer Action ---
# The real rover has FL+RL wired to one motor driver and FR+RR to another.
# This action term maps 2 NN outputs [left, right] to 4 joints by broadcasting,
# so the simulation exactly matches the physical hardware constraint.
class SkidSteerAction(ActionTerm):
    """2-input action term: [left_throttle, right_throttle].
    FL and RL always receive the same velocity command.
    FR and RR always receive the same velocity command.
    """
    cfg: "SkidSteerActionCfg"

    def __init__(self, cfg: "SkidSteerActionCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        # Resolve indices: left=[FL, RL], right=[FR, RR]
        left_ids,  _ = self._asset.find_joints(["joint_fl", "joint_rl"])
        right_ids, _ = self._asset.find_joints(["joint_fr", "joint_rr"])
        # Store combined order for a single set_joint_velocity_target call
        self._joint_ids = left_ids + right_ids  # [FL, RL, FR, RR]
        # Buffers
        self._raw_actions       = torch.zeros(env.num_envs, 2, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 2, device=env.device)

    @property
    def action_dim(self) -> int:
        return 2  # [left, right]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions.clone()
        self._processed_actions = actions * self.cfg.scale

    def apply_actions(self):
        left  = self._processed_actions[:, 0:1]  # (num_envs, 1)
        right = self._processed_actions[:, 1:2]  # (num_envs, 1)
        # Broadcast: columns = [FL, RL, FR, RR]
        vel = torch.cat([left, left, right, right], dim=1)  # (num_envs, 4)
        self._asset.set_joint_velocity_target(vel, joint_ids=self._joint_ids)


@configclass
class SkidSteerActionCfg(ActionTermCfg):
    class_type: type = SkidSteerAction
    asset_name: str = "robot"
    scale: float = 15.0


# --- 4. Environment Config ---
@configclass
class ActionsCfg:
    """2-action skid-steer: [left_throttle, right_throttle].
    Both wheels on each side are always coupled to the same command.
    """
    wheels = SkidSteerActionCfg()

@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reinforcement learning environment."""
    # Scene settings
    scene = LynxmotionSceneCfg(num_envs=16, env_spacing=4.0)
    episode_length_s = 15.0
    decimation = 4 # Skid steer often needs slightly coarser control steps (approx 30Hz-50Hz)

    observations: ObservationsCfg = ObservationsCfg()

    # Action settings
    actions = ActionsCfg()

    events = {
        "reset_robot": EventTermCfg(
            func=mdp.reset_root_state_from_terrain,
            mode="reset",
            params={
                "pose_range": {"roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-3.14, 3.14)}, 
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "asset_cfg": SceneEntityCfg("robot")
            },
        ),
        "reset_target": EventTermCfg(
            func=reset_target_state_from_terrain,
            mode="reset",
            params={
                "pose_range": {"roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}, 
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "asset_cfg": SceneEntityCfg("target"),
            },
        ),
    }

    rewards = {
        "tracking": RewardTermCfg(func=rew_distance, params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")}, weight=0.8),
        "collision": RewardTermCfg(func=rew_collision, params={"robot_cfg": SceneEntityCfg("robot")}, weight=0.2),
    }
    
    terminations = {
        "timeout": TerminationTermCfg(func=mdp.time_out),
        "success": TerminationTermCfg(func=terminate_success, params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")})
    }
