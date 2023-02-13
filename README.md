# Learned Robot Placement

Based on the research paper: **Robot Learning of Mobile Manipulation With Reachability Behavior Priors** [1] [[Paper](https://arxiv.org/abs/2203.04051)] [[Project site](https://irosalab.com/rlmmbp/)]

<p float="left">
  <img src="https://irosalab853468903.files.wordpress.com/2022/02/1mreach.gif?w=543" width="400" />
  <img src="https://irosalab853468903.files.wordpress.com/2022/02/tablemultiobj.gif?w=543" width="400" /> 
</p>

This code is meant for learning mobile manipulation behaviors through Reinforcement Learning (RL). Specifically, the RL agent learns where to place the mobile manipulator robot and whether to activate the arm of the robot for reaching/grasping at a 6D end-effector target pose.

The repository contains RL environments for the Tiago++ mobile manipulator robot and uses the NVIDIA Isaac Sim simulator (Adapted from OmniIsaacGymEnvs [2]). It also uses the proposed algorithm **Boosted Hybrid Reinforcement Learning (BHyRL)** [1] (https://github.com/iROSA-lab/mushroom-rl/blob/dev/mushroom_rl/algorithms/actor_critic/deep_actor_critic/bhyrl.py)

## Installation

__Requirements:__ The NVIDIA ISAAC Sim simulator requires a GPU with RT (RayTracing) cores. This typically means an RTX GPU. The recommended specs are provided here: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/requirements.html

- Install isaac-sim on your PC by following the procedure outlined here: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html\
**Note:** This code was tested on isaac-sim **version 2022.2.0**. 

    Troubleshooting: (common error when starting up) https://forums.developer.nvidia.com/t/since-2022-version-error-failed-to-create-change-watch-no-space-left-on-device/218198


- Follow the isaac-sim python conda environment installation at: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_python.html#advanced-running-with-anaconda\
Note that we use a modified version of the isaac-sim conda environment `isaac-sim-lrp` which needs to be used instead and is available at `learned_robot_placement/environment.yml`. Don't forget to source the `setup_conda_env.sh` script in the isaac-sim directory before running experiments. (You could also add it to the .bashrc)
- The code uses pinocchio [3] for inverse kinematics. The installation of pinoccio is known to be troublesome but the easiest way is to run `conda install pinocchio -c conda-forge` after activating the `isaac-sim-lrp ` conda environment. NOTE: If using pinocchio, disable isaac motion planning etc. because at the moment it is incompatible with pinocchio. Edit the file `isaac/ov/pkg/isaac_sim-2022.2.0/apps/omni.isaac.sim.python.kit` and comment out lines `541` to `548`.

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
    cd <this repository>
    pip install -e .
    ```

## Experiments

### Launching the experiments
- Activate the conda environment:
    ```
    conda activate isaac-sim-lrp
    ```
- source the isaac-sim conda_setup file:
    ```
    source <PATH_TO_ISAAC_SIM>/isaac_sim-2022.1.0/setup_conda_env.sh
    ```
- To test the installation, an example random policy can be run:
    ```
    python learned_robot_placement/scripts/random_policy.py
    ```
- To launch a training experiment for a simple 6D reaching task in free space, run:
    ```
    python learned_robot_placement/scripts/train_task.py task=TiagoDualReaching train=TiagoDualReachingBHyRL num_seeds=1 headless=True
    ```
- To launch a training experiment for a reaching task with a table and multiple objects (while **boosting** on a previously trained agent in free-space), run:
    ```
    python learned_robot_placement/scripts/train_task.py task=TiagoDualMultiObjFetching train=TiagoDualMultiObjFetchingBHyRL num_seeds=1 headless=True
    ```

### Configuration and command line arguments

- We use [Hydra](https://hydra.cc/docs/intro/) to manage the experiment configuration
- Common arguments for the training scripts are: `task=<TASK>` (Selects which task to use) and `train=<TRAIN>` (Selects which training config to use).
- You can check current configurations in the `/cfg` folder

For more details about the code structure, have a look at the OmniIsaacGymEnvs docs: https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/tree/main/docs

## Add-ons:

To generate sampled reachability and base-placement maps for a mobile manipulator (as visualized in the paper [1]), have a look at: https://github.com/iROSA-lab/sampled_reachability_maps

## References

[1] S. Jauhri, J. Peters and G. Chalvatzaki, "Robot Learning of Mobile Manipulation With Reachability Behavior Priors" in IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 8399-8406, July 2022, doi: 10.1109/LRA.2022.3188109

[2] https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs

[3] https://github.com/stack-of-tasks/pinocchio

[4] C. D’Eramo, D. Tateo, A. Bonarini, M. Restelli, and J. Peters, “Mushroom-rl: Simplifying reinforcement learning research,” JMLR, vol. 22, pp. 131:1–131:5, 2021

## Troubleshooting

- **"[Error] [omni.physx.plugin] PhysX error: PxRigidDynamic::setGlobalPose: pose is not valid."** This error can be **ignored** for now. Isaac-sim may have some trouble handling the set_world_pose() function for RigidPrims, but this doesn't affect the experiments.
- **"[Error] no space left on device"** https://forums.developer.nvidia.com/t/since-2022-version-error-failed-to-create-change-watch-no-space-left-on-device/218198