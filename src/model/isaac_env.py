import torch
import math
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import SceneEntityCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg, EventTermCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.envs.mdp as mdp

# --- 1. Scene Configuration ---
@configclass
class LynxmotionSceneCfg(InteractiveSceneCfg):
    """Configuration for the Lynxmotion A4WD1 scene."""
    
    # Ground Plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # THE ROBOT: 4-Wheel Skid Steer Configuration
    # Note: If you have your own A4WD1 USD, change 'usd_path' and 'joint_names' accordingly.
    # We use the Husky here as a placeholder because it has the same 4-wheel kinematics.
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/Clearpath/Husky/husky.usd", 
            activate_contact_sensors=False,
            scale=(0.3, 0.3, 0.3), # Scaling down Husky to roughly match A4WD1 size (~30cm)
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
        ),
        actuators={
            # We define one actuator group for all 4 wheels
            "all_wheels": ImplicitActuatorCfg(
                # These names must match the joints in your USD
                joint_names_expr=["front_left_wheel", "rear_left_wheel", "front_right_wheel", "rear_right_wheel"],
                effort_limit=20.0,
                velocity_limit=15.0,
                stiffness=0.0,
                damping=100.0,
            ),
        },
    )

    # Target (Red Sphere)
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(radius=0.1, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 2.0, 0.1)),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    )

    # Lidar Sensor (24 samples, 360 degrees)
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/lidar", # Ensure this attaches to the correct body link name
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, 
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0, 360),
            horizontal_fov_count=24 # Matches your NN input
        ),
        debug_vis=False, 
        mesh_prim_paths=["/World/ground"], 
    )

# --- 2. Observation Functions ---
def obs_current_pos(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[sensor_cfg.name]
    return robot.data.root_pos_w[:, :2]

def obs_target_pos(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    target = env.scene[sensor_cfg.name]
    return target.data.root_pos_w[:, :2]

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
    return -dist

def rew_success(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    dist = torch.norm(target.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    return (dist < 0.25).float() * 50.0

# --- 4. Environment Config ---
@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    scene = LynxmotionSceneCfg(num_envs=4, env_spacing=4.0)
    episode_length_s = 15.0
    decimation = 4 # Skid steer often needs slightly coarser control steps (approx 30Hz-50Hz)

    actions = {
        # We control 4 joints, but we will feed it a (N,4) tensor manually constructed from your (N,2) model output
        "wheels": mdp.JointVelocityAction(
            asset_name="robot", 
            joint_names=["front_left_wheel", "rear_left_wheel", "front_right_wheel", "rear_right_wheel"], 
            scale=10.0
        )
    }

    observations = {
        "policy": ObservationGroupCfg(
            concatenated=True,
            term_order=["pos", "target", "lidar"], 
            terms={
                "pos": ObservationTermCfg(func=obs_current_pos, params={"sensor_cfg": SceneEntityCfg("robot")}),
                "target": ObservationTermCfg(func=obs_target_pos, params={"sensor_cfg": SceneEntityCfg("target")}),
                "lidar": ObservationTermCfg(func=obs_lidar, params={"sensor_cfg": SceneEntityCfg("lidar")}),
            },
        )
    }

    events = {
        "reset_robot": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}, "asset_cfg": SceneEntityCfg("robot")},
        ),
        "reset_target": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}, "asset_cfg": SceneEntityCfg("target")},
        ),
    }

    rewards = {
        "tracking": RewardTermCfg(func=rew_distance, params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")}, weight=1.5),
        "success": RewardTermCfg(func=rew_success, params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")}, weight=1.0),
    }
    
    terminations = {
        "timeout": TerminationTermCfg(func=mdp.time_out),
    }