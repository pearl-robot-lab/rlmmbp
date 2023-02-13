# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import pinocchio as pin # Optional: Needs to be imported before SimApp to avoid dependency issues
from omni.isaac.kit import SimulationApp
import learned_robot_placement
import numpy as np
import torch
import carb
import os

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.viewer import ImageViewer

RENDER_WIDTH = 1280 # 1600
RENDER_HEIGHT = 720 # 900
RENDER_DT = 1.0/60.0 # 60 Hz

class IsaacEnvMushroom(Environment):
    """ This class provides a base interface for connecting RL policies with task implementations.
        APIs provided in this interface follow the interface in gym.Env and Mushroom Environemnt.
        This class also provides utilities for initializing simulation apps, creating the World,
        and registering a task.
    """

    def __init__(self, headless: bool, render: bool, sim_app_cfg_path: str) -> None:
        """ Initializes RL and task parameters.

        Args:
            headless (bool): Whether to run training headless.
            render (bool): Whether to run simulation rendering (if rendering using the ImageViewer or saving the renders).
        """
        # TODO: Set isaac sim config from file (if provided)
        # if sim_app_cfg_path: sim_app_cfg_path = os.path.dirname(learned_robot_placement.__file__) + sim_app_cfg_path
        self._simulation_app = SimulationApp({#"experience": sim_app_cfg_path,
                                              "headless": headless,
                                              "window_width": 1920,
                                              "window_height": 1080,
                                              "width": RENDER_WIDTH,
                                              "height": RENDER_HEIGHT})
        carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)
        self._run_sim_rendering = ((not headless) or render) # tells the simulator to also perform rendering in addition to physics
        self._render = self._run_sim_rendering

        # Optional ImageViewer
        self._viewer = ImageViewer([RENDER_WIDTH, RENDER_HEIGHT], RENDER_DT)
        self.sim_frame_count = 0

    def set_task(self, task, backend="torch", sim_params=None, init_sim=True) -> None:
        """ Creates a World object and adds Task to World. 
            Initializes and registers task to the environment interface.
            Triggers task start-up.

        Args:
            task (RLTask): The task to register to the env.
            backend (str): Backend to use for task. Can be "numpy" or "torch".
            sim_params (dict): Simulation parameters for physics settings. Defaults to None.
            init_sim (Optional[bool]): Automatically starts simulation. Defaults to True.
        """

        from omni.isaac.core.world import World

        self._device = "cpu"
        if sim_params and "use_gpu_pipeline" in sim_params:
            if sim_params["use_gpu_pipeline"]:
                self._device = sim_params["sim_device"]

        self._world = World(
            stage_units_in_meters=1.0, rendering_dt=RENDER_DT, backend=backend, sim_params=sim_params, device=self._device
        )
        self._world.add_task(task)
        self._task = task
        self._num_envs = self._task.num_envs
        # assert (self._num_envs == 1), "Mushroom Env cannot currently handle running multiple environments in parallel! Set num_envs to 1"

        self.observation_space = self._task.observation_space
        self.action_space = self._task.action_space
        self.num_states = self._task.num_states # Optional
        self.state_space = self._task.state_space # Optional
        gamma = self._task._gamma
        horizon = self._task._max_episode_length

        # Create MDP info for mushroom
        mdp_info = MDPInfo(self.observation_space, self.action_space, gamma, horizon)
        super().__init__(mdp_info)

        if init_sim:
            self._world.reset()

    def render(self) -> None:
        """ Step the simulation renderer and display task render in ImageViewer.
        """

        self._world.render()
        # Get render from task
        task_render = self._task.get_render()
        # Display
        self._viewer.display(task_render)
        return

    def get_render(self):
        """ Step the simulation renderer and return the render as per the task.
        """

        self._world.render()
        return self._task.get_render()
        
    def close(self) -> None:
        """ Closes simulation.
        """

        self._simulation_app.close()
        return

    def seed(self, seed=-1):
        """ Sets a seed. Pass in -1 for a random seed.

        Args:
            seed (int): Seed to set. Defaults to -1.
        Returns:
            seed (int): Seed that was set.
        """

        from omni.isaac.core.utils.torch.maths import set_seed

        return set_seed(seed)

    def step(self, action):
        """ Basic implementation for stepping simulation. 
            Can be overriden by inherited Env classes
            to satisfy requirements of specific RL libraries. This method passes actions to task
            for processing, steps simulation, and computes observations, rewards, and resets.

        Args:
            action (numpy.ndarray): Action from policy.
        Returns:
            observation(numpy.ndarray): observation data.
            reward(numpy.ndarray): rewards data.
            done(numpy.ndarray): reset/done data.
            info(dict): Dictionary of extra data.
        """
        # pass action to task for processing
        task_actions = torch.unsqueeze(torch.tensor(action,dtype=torch.float,device=self._device),dim=0)
        self._task.pre_physics_step(task_actions)

        # allow users to specify the control frequency through config
        for _ in range(self._task.control_frequency_inv):
            self._world.step(render=self._run_sim_rendering)
            self.sim_frame_count += 1

        obs, rews, resets, extras = self._task.post_physics_step() # buffers of obs, reward, dones and infos. Need to be squeezed

        observation = obs[0].cpu().numpy()
        reward = rews[0].cpu().item()
        done = resets[0].cpu().item()
        info = extras[0].cpu().item()

        return observation, reward, done, info

    def reset(self, state=None):
        """ Resets the task and updates observations. """
        self._task.reset()
        self._world.step(render=self._run_sim_rendering)
        observation = self._task.get_observations()[0].cpu().numpy()

        return observation

    def stop(self):
        pass

    def shutdown(self):
        pass

    @property
    def num_envs(self):
        """ Retrieves number of environments.

        Returns:
            num_envs(int): Number of environments.
        """
        return self._num_envs
