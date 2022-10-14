# Learned Robot Placement

Based on the research paper: **Robot Learning of Mobile Manipulation With Reachability Behavior Priors** [1] [[Paper](https://arxiv.org/abs/2203.04051)] [[Project site](https://irosalab.com/rlmmbp/)]

<p float="left">
  <img src="https://irosalab853468903.files.wordpress.com/2022/02/1mreach.gif?w=543" width="400" />
  <img src="https://irosalab853468903.files.wordpress.com/2022/02/3obsreach.gif?w=543" width="400" /> 
</p>

This code is meant for learning mobile manipulation behaviors through Reinforcement Learning (RL). Specifically, the RL agent learns where to place the mobile manipulator robot and whether to activate the arm of the robot for reaching/grasping at a 6D end-effector target pose.

The repository contains RL environments for the Tiago++ mobile manipulator robot and uses the NVIDIA Isaac Sim simulator (Adapted from OmniIsaacGymEnvs [2]). It also uses the proposed algorithm **Boosted Hybrid Reinforcement Learning (BHyRL)** [1] (https://github.com/iROSA-lab/mushroom-rl/blob/dev/mushroom_rl/algorithms/actor_critic/deep_actor_critic/bhyrl.py)

## Installation

__Requirements:__ The NVIDIA ISAAC Sim simulator requires a GPU with RT (RayTracing) cores. This typically means an RTX GPU. The recommended specs are provided here: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/requirements.html\

- Install isaac-sim on your PC by following the procedure outlined here: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html\
**Note:** This code was tested on isaac-sim **version 2022.1.0**
- Follow the isaac-sim python conda environment installation at: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_python.html#advanced-running-with-anaconda\
Note that we use a modified version of the isaac-sim conda environment `isaac-sim-lrp` which needs to be used instead and is available at `learned_robot_placement/environment.yml`. Don't forget to source the `setup_conda_env.sh` script in the isaac-sim directory before running experiments. (You could also add it to the .bashrc)
- The code uses pinocchio [3] for inverse kinematics. The installation of pinoccio is known to be troublesome but the easiest way is to run `conda install pinocchio -c conda-forge` after activating the `isaac-sim-lrp ` conda environment.

### Setup RL algorithm and environments
- For RL, we use the **mushroom-rl** library [4]. Install the iROSA fork of the mushroom-rl repository (https://github.com/iROSA-lab/mushroom-rl.git) within the conda environment:
    ```
    conda activate isaac-sim-lrp
    git clone https://github.com/iROSA-lab/mushroom-rl.git
    cd mushroom-rl
    pip install -e .
    ```
    (This fork contains the implementation of **Boosted Hybrid Reinforcement Learning (BHyRL)** [1] https://github.com/iROSA-lab/mushroom-rl/blob/dev/mushroom_rl/algorithms/actor_critic/deep_actor_critic/bhyrl.py)
- Finally, install this repository's python package:
    ```
    cd learned_robot_placement
    pip install -e .
    ```

## Launching experiments

- To test the installation, an example random policy can be run:
    ```
    python learned_robot_placement/scripts/random_policy.py```
- ****TODO****

For details about the code structure, have a look at the OmniIsaacGymEnvs docs: https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/tree/main/docs

## References

[1] S. Jauhri, J. Peters and G. Chalvatzaki, "Robot Learning of Mobile Manipulation With Reachability Behavior Priors" [1] in IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 8399-8406, July 2022, doi: 10.1109/LRA.2022.3188109

[2] https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs

[3] https://github.com/stack-of-tasks/pinocchio

[4] C. D’Eramo, D. Tateo, A. Bonarini, M. Restelli, and J. Peters, “Mushroom-rl: Simplifying reinforcement learning research,” JMLR, vol. 22, pp. 131:1–131:5, 2021