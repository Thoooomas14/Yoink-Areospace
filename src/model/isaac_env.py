import torch
import math
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import SceneEntityCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg, EventTermCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfRandomUniformTerrainCfg, HfDiscreteObstaclesTerrainCfg
import isaaclab.envs.mdp as mdp
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
            border_width=5.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.001,
            sub_terrains={
                "sand": HfRandomUniformTerrainCfg(
                    proportion=1.0, noise_range=(0.0, 0.01), noise_step=0.001, downsampled_scale=0.1
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
            asset_path=f"{os.path.abspath(os.path.join(os.path.dirname(__file__), 'lynx.urdf'))}", 
            activate_contact_sensors=False,
            make_instanceable=False,
            fix_base=False,
            joint_drive=UrdfFileCfg.JointDriveCfg(
                gains=UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0)
            )
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.2), # Height to drop from slightly
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["joint_fl", "joint_fr", "joint_rl", "joint_rr"],
                effort_limit=20.0,
                velocity_limit=15.0,
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
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 2.0, 0.2)),
    )

    # Obstacles (Cylinders)
    obstacle_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_1",
        spawn=sim_utils.CylinderCfg(
            radius=0.15, height=0.4, axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            visual_material=sim_utils.MdlFileCfg(mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Natural/Dirt.mdl", project_uvw=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 1.0, 0.2)),
    )
    obstacle_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_2",
        spawn=sim_utils.CylinderCfg(
            radius=0.25, height=0.4, axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            visual_material=sim_utils.MdlFileCfg(mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Natural/Dirt.mdl", project_uvw=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0, 1.0, 0.2)),
    )
    obstacle_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_3",
        spawn=sim_utils.CylinderCfg(
            radius=0.20, height=0.4, axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            visual_material=sim_utils.MdlFileCfg(mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Natural/Dirt.mdl", project_uvw=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.0, 0.2)),
    )

    # Lidar Sensor (25 samples, 360 degrees)
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.10, 0.06, 0.18)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, 
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0, 360),
            horizontal_res=14.4 # Matches your NN input of 25 rays (360/14.4 = 25)
        ),
        debug_vis=False, 
        mesh_prim_paths=["/World"], 
    )

# --- 2. Observation Functions ---
def obs_current_pos(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[sensor_cfg.name]
    # Return robot's local velocity [Forward Speed, Yaw Rate] instead of position
    # This helps the neural net know how fast it is currently driving/turning!
    lin_vel_x = robot.data.root_lin_vel_b[:, 0]  # Forward velocity in body frame
    ang_vel_z = robot.data.root_ang_vel_b[:, 2]  # Yaw rate in body frame
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
    
    # Handle misses (where distance might be huge or 0 depending on config)
    # Usually we clamp them to max range
    return dists

# --- 3. Rewards ---
def rew_distance(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    dist = torch.norm(target.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    
    # Smooth continuous reward (closer is better, max 1.0)
    return 1.0 / (1.0 + dist)

def rew_success(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    dist = torch.norm(target.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    
    # Continuous bonus for being very close
    return torch.where(dist < 0.4, 3.0, 0.0)

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
        pos = ObservationTermCfg(func=obs_current_pos, params={"sensor_cfg": SceneEntityCfg("robot")})
        target = ObservationTermCfg(func=obs_target_pos, params={"sensor_cfg": SceneEntityCfg("target")})
        lidar = ObservationTermCfg(func=obs_lidar, params={"sensor_cfg": SceneEntityCfg("lidar")})

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


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
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}, 
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "asset_cfg": SceneEntityCfg("robot")
            },
        ),
        "reset_target": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}, 
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "asset_cfg": SceneEntityCfg("target")
            },
        ),
        "reset_obstacles_1": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-4.0, 4.0), "y": (-4.0, 4.0)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "asset_cfg": SceneEntityCfg("obstacle_1")
            },
        ),
        "reset_obstacles_2": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-4.0, 4.0), "y": (-4.0, 4.0)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "asset_cfg": SceneEntityCfg("obstacle_2")
            },
        ),
        "reset_obstacles_3": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-4.0, 4.0), "y": (-4.0, 4.0)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "asset_cfg": SceneEntityCfg("obstacle_3")
            },
        ),
    }

    rewards = {
        "tracking": RewardTermCfg(func=rew_distance, params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")}, weight=2.0),
        "alignment": RewardTermCfg(func=rew_alignment, params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")}, weight=0.5),
        "success": RewardTermCfg(func=rew_success, params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")}, weight=2.0),
    }
    
    terminations = {
        "timeout": TerminationTermCfg(func=mdp.time_out),
    }