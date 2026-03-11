import torch
import math
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import RayCasterCfg, patterns, ImuCfg
from isaaclab.managers import SceneEntityCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg, EventTermCfg
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
            size=(10.0, 10.0),
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
                    num_obstacles=20,
                    platform_width=1.5,
                    flat_patch_sampling={
                        "init_pos": FlatPatchSamplingCfg(
                            num_patches=100,
                            patch_radius=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            max_height_diff=0.1,
                            x_range=(-2.0, 2.0),
                            y_range=(-2.0, 2.0),
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
            pos=(0.0, 0.0, 0.05), # Ground level spawn to prevent dropping frames
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["joint_fl", "joint_fr", "joint_rl", "joint_rr"],
                effort_limit_sim=20.0,
                velocity_limit_sim=15.0,
                stiffness=0.0,
                damping=100.0,
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

    # Lidar Sensor (25 samples, 360 degrees)
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.10, 0.06, 0.18)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, 
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0, 360),
            horizontal_res=14.4, # Due to torch.linspace, 14.4 forces 25 points which yields exactly 24 rays spaced by 15.0 deg
        ),
        debug_vis=True, 
        mesh_prim_paths=["/World/ground"], 
    )

    # Arduino Uno WiFi IMU Sensor
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
    )

# --- 2. Observation Functions ---
def obs_current_pos(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    imu = env.scene[sensor_cfg.name]
    # Use the simulated IMU sensor's onboard gyros to measure our forward and yaw velocities!
    # This prepares the NN to natively accept the Arduino IMU's data stream in the real world.
    lin_vel_x = imu.data.lin_vel_b[:, 0]  # Forward velocity in local frame
    ang_vel_z = imu.data.ang_vel_b[:, 2]  # Yaw rate in local frame
    return torch.stack([lin_vel_x, ang_vel_z], dim=1)

def obs_target_pos(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    target = env.scene[sensor_cfg.name]
    robot = env.scene["robot"]
    
    # 1. Get relative vector (Target - Robot) in World Frame
    target_vec_w = target.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    
    # 2. Get Robot's Forward Vector in World Frame
    quat = robot.data.root_quat_w
    quat_w, quat_x, quat_y, quat_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    forward_x = 1.0 - 2.0 * (quat_y**2 + quat_z**2)
    forward_y = 2.0 * (quat_x * quat_y + quat_w * quat_z)
    
    # 3. Project Target Vector onto Robot's Local Axes (Ego-Centric Frame)
    # Dot product with Forward Vector = Local X (Distance straight ahead)
    local_x = target_vec_w[:, 0] * forward_x + target_vec_w[:, 1] * forward_y
    # Dot product with Left Vector (-y, x) = Local Y (Distance to the left/right)
    local_y = target_vec_w[:, 0] * (-forward_y) + target_vec_w[:, 1] * forward_x
    
    # Return Target Position purely relative to the robot's nose!
    return torch.stack([local_x, local_y], dim=1)

def obs_lidar(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    lidar = env.scene[sensor_cfg.name]
    # ray_hits_w is [num_envs, num_rays, 3], we calculate norm for distance
    # Or simpler: use ray_hits_w and subtract origin, but usually the 'distance' is derived or we assume hit.
    # For RayCaster, Isaac Lab usually provides 'data.ray_hits_w'. 
    # We compute distance manually to be safe:
    origins = lidar.data.pos_w # [num_envs, 3]
    hits = lidar.data.ray_hits_w # [num_envs, 24, 3]
    
    # Calculate Euclidean distance
    # Expand origins to [num_envs, 1, 3] to broadcast
    dists = torch.norm(hits - origins.unsqueeze(1), dim=-1)
    
    # Handle misses (Isaac Lab puts missed rays at max_distance, usually 1e6 meters out)
    # This blows up Neural Network weights instantly.
    # We clip it to 10.0 meters and normalize to [0, 1]
    dists = torch.clamp(dists, 0.0, 10.0)
    normalized_dists = dists / 10.0
    
    return normalized_dists

# --- 3. Rewards ---
def rew_distance(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    
    # Vector from robot to target
    target_vec_w = target.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    target_dist = torch.norm(target_vec_w, dim=1)
    
    # Unit direction vector pointing perfectly at target
    target_dir_w = target_vec_w / (target_dist.unsqueeze(1) + 1e-6)
    
    # Robot's actual linear velocity in the world frame [x, y]
    robot_vel_w = robot.data.root_lin_vel_w[:, :2]
    
    # Dot product: This IS mathematically the rate at which distance to the target is decreasing!
    vel_towards_target = torch.sum(robot_vel_w * target_dir_w, dim=1)
    
    # Get Robot's forward direction to check if it's facing the target
    quat = robot.data.root_quat_w
    quat_w, quat_x, quat_y, quat_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    forward_x = 1.0 - 2.0 * (quat_y**2 + quat_z**2)
    forward_y = 2.0 * (quat_x * quat_y + quat_w * quat_z)
    robot_dir = torch.stack([forward_x, forward_y], dim=1)
    
    alignment = torch.sum(target_dir_w * robot_dir, dim=1)
    
    # Only reward the robot for decreasing the distance IF its nose is pointed within ~60 degrees of the target.
    # If it tries to spiral at 85 degrees, alignment is ~0.08, so it earns $0.0 for the distance decrease!
    return torch.where(alignment > 0.5, vel_towards_target, 0.0)

def rew_success(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    
    # Calculate horizontal distance to target
    dist = torch.norm(target.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    
    # Calculate rover's linear speed
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    
    # 1. Proximity Gradient: Scale reward linearly starting from 0.4m away, reaching max 1.0 at 0.0m
    dist_clamped = torch.clamp(dist, 0.0, 0.4)
    proximity_reward = (0.4 - dist_clamped) / 0.4
    
    # 2. Stationary Bonus: Massive 2.0 multiplier bonus if the robot hits the brakes (< 0.1 m/s) while parked inside target area
    in_zone = dist < 0.1
    is_still = speed < 0.1
    stationary_bonus = torch.where(in_zone & is_still, 2.0, 0.0)
    
    # Return gradient reward + stationary bonus
    return proximity_reward + stationary_bonus

def rew_collision(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    # Use Lidar to calculate proximity penalty based on actual perception!
    lidar = env.scene["lidar"]
    origins = lidar.data.pos_w
    hits = lidar.data.ray_hits_w
    dists = torch.norm(hits - origins.unsqueeze(1), dim=-1)
    dists = torch.clamp(dists, 0.0, 10.0)
    
    # Minimum distance sensed on any ray
    min_dist = torch.min(dists, dim=1)[0]
    
    # surface_dist is distance from robot chassis edge (approx 0.2m radius) to rock surface
    surface_dist = torch.clamp(min_dist - 0.2, 0.0, 10.0)
    
    # Starts penalizing exactly when breaching the 30cm safety bubble.
    # Linearly scales up to a 1.0 penalty if it physically hits the rock (0.0m)
    proximity_penalty = (0.3 - surface_dist) / 0.3
    
    return torch.clamp(proximity_penalty, 0.0, 1.0)

def rew_alignment(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    
    # Vector from robot to target
    target_vec = target.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    target_dist = torch.norm(target_vec, dim=1)
    target_dir = target_vec / (target_dist.unsqueeze(1) + 1e-5)
    
    # Robot forward direction (assuming X is forward in local frame)
    # We can get this from the quaternion or just use root_forward_w if available
    # Isaac Lab's RigidObjectData usually has root_quat_w
    # A simple way to get heading in 2D from quat (w, x, y, z):
    # heading = atan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    # Or more directly, transform the local +X vector (1, 0, 0) by the quaternion
    # Isaac Lab provides `data.projected_gravity_b` or similar, but we can compute forward:
    
    # We can use the math utility or manually compute.
    # To keep dependencies minimal:
    quat = robot.data.root_quat_w
    # q = [w, x, y, z] -> forward vector is primarily the first column of the rotation matrix
    # vx = 1 - 2*(y^2 + z^2), vy = 2*(x*y + w*z)
    quat_w, quat_x, quat_y, quat_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    forward_x = 1.0 - 2.0 * (quat_y**2 + quat_z**2)
    forward_y = 2.0 * (quat_x * quat_y + quat_w * quat_z)
    
    robot_dir = torch.stack([forward_x, forward_y], dim=1)
    
    # Dot product gives the cosine of the angle between them
    alignment = torch.sum(target_dir * robot_dir, dim=1)
    
    # Reward positive alignment (facing target)
    return alignment

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

# --- 4. Environment Config ---
@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    wheels: mdp.JointVelocityActionCfg = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["joint_fl", "joint_fr", "joint_rl", "joint_rr"],
        scale=10.0,
    )

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
        "tracking": RewardTermCfg(func=rew_distance, params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")}, weight=3.0),
        "alignment": RewardTermCfg(func=rew_alignment, params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")}, weight=0.0),
        "success": RewardTermCfg(func=rew_success, params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")}, weight=5.0),
        "collision": RewardTermCfg(func=rew_collision, params={"robot_cfg": SceneEntityCfg("robot")}, weight=-2.0),
    }
    
    terminations = {
        "timeout": TerminationTermCfg(func=mdp.time_out),
    }