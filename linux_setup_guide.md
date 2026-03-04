# Isaac Sim + Lab Linux/WSL Setup Guide
This guide outlines the steps needed to migrate the `yoink` project to a native Ubuntu 22.04 VM (or Windows Subsystem for Linux), where Isaac Sim can render the Mars terrain properly without `omni.kit` memory violation bugs.

## Prerequisites
- A virtual machine or WSL2 instance running **Ubuntu 20.04** or **22.04**.
- An NVIDIA GPU with Linux drivers (`>= 535`) installed.
- **Miniconda / Anaconda** installed.
- **Git** LFS installed (`sudo apt install git-lfs`).

---

## 1. Create the Conda Environment
Instead of using the GUI Omniverse Launcher, we can install Isaac Sim directly via pip into a fresh conda environment. This is the officially recommended approach for headless cloud VMs.

1. In your WSL/Ubuntu terminal, create and activate a new Python 3.11 environment:
   ```bash
   conda create -n yoink python=3.11 -y
   conda activate yoink
   ```

---

## 2. Install Isaac Sim 5.1 (Terminal-Only)
NVIDIA now hosts Isaac Sim on their PyPI server, so you can pull the entire 10GB engine directly through `pip` without any GUI.

1. Install the Isaac Sim pip package:
   ```bash
   pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
   ```
2. *(Optional)* Verify the physics engine works:
   ```bash
   python -c "from isaacsim import SimulationApp; SimulationApp({'headless': True}).close()"
   ```

---

## 3. Install Isaac Lab
Isaac Lab contains the Reinforcement Learning libraries (`ManagerBasedRLEnv`, etc.) we used to build the procedural terrain logic.

1. Clone Isaac Lab:
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   ```
2. Run the Isaac Lab installer. Because we installed Isaac Sim via `pip`, the installer will automatically detect it:
   ```bash
   ./isaaclab.sh --install
   ```

---

## 4. Install Project Requirements
Now that the core simulator and RL frameworks are installed, retrieve your code.

1. Clone this project repository to your VM:
   ```bash
   git clone <your-repo-url>
   cd Yoink-Areospace
   ```
2. Install the remaining simple dependencies like Tensorboard from the `environment.yml`:
   ```bash
   conda env update -n yoink -f environment.yml
   ```
   *(If `conda env update` throws conflicts from Windows packages, simply run `pip install torch torchvision torchaudio tensorboard`.)*

---

## 4. Run the Training!
Test it exactly as we did locally. Since Linux uses a different Vulkan backend (X11/Wayland), the memory-access crashes will be bypassed.

1. First, verify the terrain generation headlessly (to test physics):
   ```bash
   conda run -n yoink python src/model/train.py --headless --num_envs 50
   ```
2. Or, run natively to visualize the 20x20m discrete rocks without crashing:
   ```bash
   conda run -n yoink python src/model/train.py
   ```
