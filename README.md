# Yoink-Aerospace
**Mechatronics and Robotics Design (MREN 203) Course Project**

Yoink-Aerospace is an autonomous rover project that bridges realistic physics simulation with real-world hardware. It uses **Isaac Sim / Isaac Lab** to train a reinforcement learning agent capable of navigating Mars-like terrain, and **ROS 2 FastDDS** to synchronously deploy that policy onto a physical Lynxmotion A4WD1 rover via a digital twin setup.

<img src="Sim_visual.png" width="900">
<video width="900" controls>
  <source src="/MREN203_short.mp4" type="video/mp4">
</video>

---

## 📂 Repository Organization

This repository is split into several crucial components:

- **`src/model/`** 
  Contains the Isaac Sim / Isaac Lab environment, training scripts, and inference digital twin.
  - `train.py`: The main reinforcement learning training script using Stable Baselines 3.
  - `eval.py`: Loads a trained policy and tests it in simulation with a custom visual debugging UI.
  - `infrence.py`: The "Digital Twin" node. Runs Isaac Sim, listens to ROS 2 topics from the real rover, updates the digital twin, and runs the policy to compute and publish drive commands.
  - `isaac_env.py`: The core Isaac Lab environment definition, configuring the robot, terrain, sensors (LiDAR/IMU), and reward weightings.
  - `yoink.py`: SB3 PPO model configuration.

- **`rover_bringup/`**
  The ROS 2 package designed to run on the rover's onboard computer (e.g., Raspberry Pi).
  - `serial_bridge_node.py`: Bridges ROS 2 with the Arduino microcontroller over Serial, sending motor commands and receiving encoder/IMU telemetry.
  - `lidar_bridge_node.py`: Processes and publishes 2D LiDAR data to the ROS 2 network.

- **`src/ArduinoControler/` & `src/ArduinoTest/`**
  C++ PlatformIO projects for the onboard Arduino. Handles the low-level PID motor control, encoder tick counting, and IMU data formatting before shipping it out over Serial.

---

## 🛠️ Installation & Setup

### 1. Isaac Sim & Isaac Lab (Linux / Headless Server)
Because training reinforcement learning policies in visually rich environments can be extremely resource-intensive, we recommend training on a Linux machine or cloud VM (Ubuntu 20.04/22.04 with NVIDIA drivers).

**👉 See the dedicated [Linux Setup Guide](linux_setup_guide.md) for step-by-step instructions on setting up Isaac Sim 5.1 via pip, configuring Isaac Lab, and connecting via WebRTC livestreaming.**

### 2. Rover Hardware Environment
For the physical rover (Ubuntu + ROS 2 Humble/Iron):
1. Navigate to your ROS 2 workspace `src` folder and clone this repo.
2. Build the ROS 2 package:
   ```bash
   colcon build --packages-select rover_bringup
   source install/setup.bash
   ```
3. For the Arduino code, use **PlatformIO** (VS Code extension or CLI) to compile and upload `src/ArduinoControler` to your board.

---

## 🚀 Running Critical Scripts

### 1. Training the RL Model
To train the rover to navigate the procedural terrain:
```bash
conda run -n yoink python -m src.model.train --num_envs 50
```
**Key Arguments:**
- `--num_envs`: Number of parallel robots to simulate (Default: 16).
- `--curriculum`: Enable automatic reward weight shifting during training.
- `--checkpoint`: Path to an existing `.zip` checkpoint to resume training.
- `--headless`: Run without a UI (recommended for cloud training).

### 2. Evaluating the Policy
To test a trained checkpoint in a visual debugging environment:
```bash
conda run -n yoink python -m src.model.eval --num_envs 4 --checkpoint src/model/checkpoints/checkpoint_progress.zip
```
This launches a custom debugger window showing exactly what the neural network "sees" (Virtual LiDAR radar, IMU stats, target distance) and what it is outputting.

### 3. Digital Twin / Inference
To run the live synchronization and inference against the physical hardware:
```bash
conda run -n yoink python -m src.model.infrence --domain_id 15
```
**How it works:**
1. The script launches Isaac Sim and loads the ROS 2 Bridge extension.
2. It waits for telemetry and LiDAR data from the physical rover on the `/features` topic.
3. Upon receiving data, it instantly snaps the virtual rover to match the real rover's pose.
4. It computes the distance to the target, feeds the bundled observations into the SB3 Neural Network, and publishes the `v, omega` Twist command back over the `/policy_action` topic.

*Ensure the `ROS_DOMAIN_ID` matches your rover's configuration!*

---

## 📡 ROS 2 Architecture

The system utilizes a decentralized architecture:

1. **Physical Rover (Raspberry Pi)**
   - Runs `lidar_bridge_node` generating spatial data.
   - Runs `serial_bridge_node` executing velocity commands and publishing odometry.
2. **Base Station (PC/Cloud)**
   - Runs `infrence.py` (The Digital Twin).
   - Subscribes to the rover's state and sensory data.
   - Publishes intelligent navigation commands.

By coupling the system this way, the computationally heavy AI inference is offloaded from the low-power physical rover, and we can visualize the real-world operation inside the Omniverse simulator in real-time.
