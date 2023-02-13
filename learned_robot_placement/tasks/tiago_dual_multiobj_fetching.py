# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np
from learned_robot_placement.tasks.base.rl_task import RLTask
from learned_robot_placement.handlers.tiagodualWBhandler import TiagoDualWBHandler
from omni.isaac.core.objects.cone import VisualCone
from omni.isaac.core.prims import GeometryPrimView
from learned_robot_placement.tasks.utils.pinoc_utils import PinTiagoIKSolver # For IK
from learned_robot_placement.tasks.utils import scene_utils
from omni.isaac.isaac_sensor import _isaac_sensor

# from omni.isaac.core.utils.prims import get_prim_at_path
# from omni.isaac.core.utils.prims import create_prim
# from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats, quat_diff_rad
from scipy.spatial.transform import Rotation

# Base placement environment for fetching a target object among clutter
class TiagoDualMultiObjFetchingTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._device = self._cfg["sim_device"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        
        self._gamma = self._task_cfg["env"]["gamma"]
        self._max_episode_length = self._task_cfg["env"]["horizon"]
        
        self._randomize_robot_on_reset = self._task_cfg["env"]["randomize_robot_on_reset"]

        # Get dt for integrating velocity commands and checking limit violations
        self._dt = torch.tensor(self._sim_config.task_config["sim"]["dt"]*self._sim_config.task_config["env"]["controlFrequencyInv"],device=self._device)

        # Environment object settings: (reset() randomizes the environment)
        self._obstacle_names = ["mammut", "godishus"] # ShapeNet models in usd format
        self._tabular_obstacle_mask = [True, True] # Mask to denote which objects are tabular (i.e. grasp objects can be placed on them)
        self._grasp_obj_names = ["004_sugar_box", "008_pudding_box", "010_potted_meat_can", "061_foam_brick"] # YCB models in usd format
        self._num_obstacles = min(self._task_cfg["env"]["num_obstacles"],len(self._obstacle_names))
        self._num_grasp_objs = min(self._task_cfg["env"]["num_grasp_objects"],len(self._grasp_obj_names))
        self._obj_states = torch.zeros((6*(self._num_obstacles+self._num_grasp_objs-1),self._num_envs),device=self._device) # All grasp objs except the target object will be used in obj state (BBox)
        self._obstacles = []
        self._obstacles_dimensions = []
        self._grasp_objs = []
        self._grasp_objs_dimensions = []
        #  Contact sensor interface for collision detection:
        self._contact_sensor_interface = _isaac_sensor.acquire_contact_sensor_interface()

        # Choose num_obs and num_actions based on task
        # 6D goal/target object grasp pose + 6D bbox for each obstacle in the room. All grasp objs except the target object will be used in obj state
        # (3 pos + 4 quat + 6*(n-1)= 7 + )
        self._num_observations = 7 + len(self._obj_states)
        self._move_group = self._task_cfg["env"]["move_group"]
        self._use_torso = self._task_cfg["env"]["use_torso"]
        # Position control. Actions are base SE2 pose (3) and discrete arm activation (2)
        self._num_actions = self._task_cfg["env"]["continous_actions"] + self._task_cfg["env"]["discrete_actions"]
        # env specific limits
        self._world_xy_radius = self._task_cfg["env"]["world_xy_radius"]
        self._action_xy_radius = self._task_cfg["env"]["action_xy_radius"]
        self._action_ang_lim = self._task_cfg["env"]["action_ang_lim"]
        # self.max_arm_vel = torch.tensor(self._task_cfg["env"]["max_rot_vel"], device=self._device)
        # self.max_rot_vel = torch.tensor(self._task_cfg["env"]["max_rot_vel"], device=self._device)
        # self.max_base_xy_vel = torch.tensor(self._task_cfg["env"]["max_base_xy_vel"], device=self._device)
        
        # End-effector reaching settings
        self._goal_pos_threshold = self._task_cfg["env"]["goal_pos_thresh"]
        self._goal_ang_threshold = self._task_cfg["env"]["goal_ang_thresh"]
        # For now, setting dummy goal:
        self._goals = torch.hstack((torch.tensor([[0.8,0.0,0.4+0.15]]),euler_angles_to_quats(torch.tensor([[0.19635, 1.375, 0.19635]]),device=self._device)))[0].repeat(self.num_envs,1)
        self._goal_tf = torch.zeros((4,4),device=self._device)
        self._goal_tf[:3,:3] = torch.tensor(Rotation.from_quat(np.array([self._goals[0,3+1],self._goals[0,3+2],self._goals[0,3+3],self._goals[0,3]])).as_matrix(),dtype=float,device=self._device) # Quaternion in scalar last format!!!
        self._goal_tf[:,-1] = torch.tensor([self._goals[0,0], self._goals[0,1], self._goals[0,2], 1.0],device=self._device) # x,y,z,1
        self._curr_goal_tf = self._goal_tf.clone()
        self._goals_xy_dist = torch.linalg.norm(self._goals[:,0:2],dim=1)  # distance from origin

        # Reward settings
        self._reward_success = self._task_cfg["env"]["reward_success"]
        self._reward_dist_weight = self._task_cfg["env"]["reward_dist_weight"]
        self._reward_noIK = self._task_cfg["env"]["reward_noIK"]
        # self._reward_timeout = self._task_cfg["env"]["reward_timeout"]
        self._reward_collision = self._task_cfg["env"]["reward_collision"]
        self._collided = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._ik_fails = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._is_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        # IK solver
        self._ik_solver = PinTiagoIKSolver(move_group=self._move_group, include_torso=self._use_torso, include_base=False, max_rot_vel=100.0) # No max rot vel
        # Handler for Tiago
        self.tiago_handler = TiagoDualWBHandler(move_group=self._move_group, use_torso=self._use_torso, sim_config=self._sim_config, num_envs=self._num_envs, device=self._device)

        RLTask.__init__(self, name, env)

    def set_up_scene(self, scene) -> None:
        import omni
        self.tiago_handler.get_robot()
        # Spawn obstacles (from ShapeNet usd models):
        for i in range(self._num_obstacles):
            obst = scene_utils.spawn_obstacle(name=self._obstacle_names[i], prim_path=self.tiago_handler.default_zero_env_path, device=self._device)
            self._obstacles.append(obst) # Add to list of obstacles (Geometry Prims)
            # Optional: Add contact sensors for collision detection. Covers whole body by default
            omni.kit.commands.execute("IsaacSensorCreateContactSensor", path="/Contact_Sensor", sensor_period=float(self._sim_config.task_config["sim"]["dt"]),
                parent=obst.prim_path)
        # Spawn grasp objs (from YCB usd models):
        for i in range(self._num_grasp_objs):
            grasp_obj = scene_utils.spawn_grasp_object(name=self._grasp_obj_names[i], prim_path=self.tiago_handler.default_zero_env_path, device=self._device)
            self._grasp_objs.append(grasp_obj) # Add to list of grasp objects (Rigid Prims)
            # Optional: Add contact sensors for collision detection. Covers whole body by default
            omni.kit.commands.execute("IsaacSensorCreateContactSensor", path="/Contact_Sensor", sensor_period=float(self._sim_config.task_config["sim"]["dt"]),
                parent=grasp_obj.prim_path)
        # Goal visualizer
        goal_viz = VisualCone(prim_path=self.tiago_handler.default_zero_env_path+"/goal",
                                radius=0.05,height=0.05,color=np.array([1.0,0.0,0.0]))
        super().set_up_scene(scene)
        self._robots = self.tiago_handler.create_articulation_view()
        scene.add(self._robots)
        self._goal_vizs = GeometryPrimView(prim_paths_expr="/World/envs/.*/goal",name="goal_viz")
        scene.add(self._goal_vizs)
        # Enable object axis-aligned bounding box computations
        scene.enable_bounding_boxes_computations()
        # Add spawned objects to scene registry and store their bounding boxes:
        for obst in self._obstacles:
            scene.add(obst)
            self._obstacles_dimensions.append(scene.compute_object_AABB(obst.name)) # Axis aligned bounding box used as dimensions
        for grasp_obj in self._grasp_objs:
            scene.add(grasp_obj)
            self._grasp_objs_dimensions.append(scene.compute_object_AABB(grasp_obj.name)) # Axis aligned bounding box used as dimensions
        # Optional viewport for rendering in a separate viewer
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        self.viewport_window = omni.kit.viewport_legacy.get_default_viewport_window()
        self.sd_helper = SyntheticDataHelper()
        self.sd_helper.initialize(sensor_names=["rgb"], viewport=self.viewport_window)

    def post_reset(self):
        # reset that takes place when the isaac world is reset (typically happens only once)
        self.tiago_handler.post_reset()

    def get_observations(self):
        # Handle any pending resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        # # Get robot observations
        # robot_joint_pos = self.tiago_handler.get_robot_obs()
        # Fill observation buffer
        # Goal: 3D pos + rot_quaternion (3+4=7)
        curr_goal_pos = self._curr_goal_tf[0:3,3].unsqueeze(dim=0)
        curr_goal_quat = torch.tensor(Rotation.from_matrix(self._curr_goal_tf[:3,:3]).as_quat()[[3, 0, 1, 2]],dtype=torch.float,device=self._device).unsqueeze(dim=0)
        # oriented bounding boxes of objects
        curr_bboxes_flattened = self._curr_obj_bboxes.flatten().unsqueeze(dim=0)
        self.obs_buf = torch.hstack((curr_goal_pos,curr_goal_quat,curr_bboxes_flattened))
        # TODO: Scale or normalize robot observations as per env
        return self.obs_buf

    def get_render(self):
        # Get ground truth viewport rgb image
        gt = self.sd_helper.get_groundtruth(
            ["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
        )
        return np.array(gt["rgb"][:, :, :3])
    
    def pre_physics_step(self, actions) -> None:
        # actions (num_envs, num_action)
        # Handle resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # Scale actions and convert polar co-ordinates r-phi to x-y
        # NOTE: actions shape has to match the move_group selected
        r_scaled = actions[:,0]*self._action_xy_radius
        phi_scaled = actions[:,1]*self._action_ang_lim
        x_scaled = r_scaled*torch.cos(phi_scaled)
        y_scaled = r_scaled*torch.sin(phi_scaled)
        theta_scaled = actions[:,2]*self._action_ang_lim
        
        # NOTE: Actions are in robot frame but the handler is in world frame!
        # Get current base positions
        base_joint_pos = self.tiago_handler.get_robot_obs()[:,:3] # First three are always base positions
        
        base_tf = torch.zeros((4,4),device=self._device)
        base_tf[:2,:2] = torch.tensor([[torch.cos(base_joint_pos[0,2]), -torch.sin(base_joint_pos[0,2])],[torch.sin(base_joint_pos[0,2]), torch.cos(base_joint_pos[0,2])]]) # rotation about z axis
        base_tf[2,2] = 1.0 # No rotation here
        base_tf[:,-1] = torch.tensor([base_joint_pos[0,0], base_joint_pos[0,1], 0.0, 1.0]) # x,y,z,1

        # Transform actions to world frame and apply to base
        action_tf = torch.zeros((4,4),device=self._device)
        action_tf[:2,:2] = torch.tensor([[torch.cos(theta_scaled[0]), -torch.sin(theta_scaled[0])],[torch.sin(theta_scaled[0]), torch.cos(theta_scaled[0])]])
        action_tf[2,2] = 1.0 # No rotation here
        action_tf[:,-1] = torch.tensor([x_scaled[0], y_scaled[0], 0.0, 1.0]) # x,y,z,1

        new_base_tf = torch.matmul(base_tf,action_tf)
        new_base_xy = new_base_tf[0:2,3].unsqueeze(dim=0)
        new_base_theta = torch.arctan2(new_base_tf[1,0],new_base_tf[0,0]).unsqueeze(dim=0).unsqueeze(dim=0)
        
        # Move base
        self.tiago_handler.set_base_positions(torch.hstack((new_base_xy,new_base_theta)))
        
        # Transform goal to robot frame
        inv_base_tf = torch.linalg.inv(new_base_tf)
        self._curr_goal_tf = torch.matmul(inv_base_tf,self._goal_tf)
        # Transform all other object oriented bounding boxes to robot frame
        for obj_num in range(self._num_obstacles+self._num_grasp_objs-1):
            min_xy_vertex = torch.hstack(( self._obj_bboxes[obj_num,0:2], torch.tensor([0.0, 1.0],device=self._device) )).T
            max_xy_vertex = torch.hstack(( self._obj_bboxes[obj_num,2:4], torch.tensor([0.0, 1.0],device=self._device) )).T
            new_min_xy_vertex = torch.matmul(inv_base_tf,min_xy_vertex)[0:2].T.squeeze()
            new_max_xy_vertex = torch.matmul(inv_base_tf,max_xy_vertex)[0:2].T.squeeze()
            self._curr_obj_bboxes[obj_num,0:4] = torch.hstack(( new_min_xy_vertex, new_max_xy_vertex ))
            self._curr_obj_bboxes[obj_num,5] -= theta_scaled[0] # new theta

        # Discrete Arm action:
        self._ik_fails[0] = 0
        if(actions[0,3] > actions[0,4]): # This is the arm decision variable TODO: Parallelize
            # Compute IK to self._curr_goal_tf
            curr_goal_pos = self._curr_goal_tf[0:3,3]
            curr_goal_quat = Rotation.from_matrix(self._curr_goal_tf[:3,:3]).as_quat()[[3, 0, 1, 2]]
            success, ik_positions = self._ik_solver.solve_ik_pos_tiago(des_pos=curr_goal_pos.cpu().numpy(), des_quat=curr_goal_quat,
                                    pos_threshold=self._goal_pos_threshold, angle_threshold=self._goal_ang_threshold, verbose=False)
            if success:
                self._is_success[0] = 1 # Can be used for reward, termination
                # set upper body positions
                self.tiago_handler.set_upper_body_positions(jnt_positions=torch.tensor(np.array([ik_positions]),dtype=torch.float,device=self._device))
            else:
                self._ik_fails[0] = 1 # Can be used for reward

    def reset_idx(self, env_ids):
        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        # reset dof values
        self.tiago_handler.reset(indices,randomize=self._randomize_robot_on_reset)
        # reset the scene objects (randomize), get target end-effector goal/grasp as well as oriented bounding boxes of all other objects
        self._curr_grasp_obj, self._goals[env_ids], self._obj_bboxes = scene_utils.setup_tabular_scene(
                                self, self._obstacles, self._tabular_obstacle_mask[0:self._num_obstacles], self._grasp_objs,
                                self._obstacles_dimensions, self._grasp_objs_dimensions, self._world_xy_radius, self._device)
        self._curr_obj_bboxes = self._obj_bboxes.clone()
        # self._goals[env_ids] = torch.hstack((goals_sample[:,:3],euler_angles_to_quats(goals_sample[:,3:6],device=self._device)))
        
        self._goal_tf = torch.zeros((4,4),device=self._device)
        goal_rot = Rotation.from_quat(np.array([self._goals[0,3+1],self._goals[0,3+2],self._goals[0,3+3],self._goals[0,3]])) # Quaternion in scalar last format!!!
        self._goal_tf[:3,:3] = torch.tensor(goal_rot.as_matrix(),dtype=float,device=self._device)
        self._goal_tf[:,-1] = torch.tensor([self._goals[0,0], self._goals[0,1], self._goals[0,2], 1.0],device=self._device) # x,y,z,1
        self._curr_goal_tf = self._goal_tf.clone()
        self._goals_xy_dist = torch.linalg.norm(self._goals[:,0:2],dim=1) # distance from origin
        # Pitch visualizer by 90 degrees for aesthetics
        goal_viz_rot = goal_rot * Rotation.from_euler("xyz", [0,np.pi/2.0,0])
        self._goal_vizs.set_world_poses(indices=indices,positions=self._goals[env_ids,:3],
                orientations=torch.tensor(goal_viz_rot.as_quat()[[3, 0, 1, 2]],device=self._device).unsqueeze(dim=0))

        # bookkeeping
        self._is_success[env_ids] = 0
        self._ik_fails[env_ids] = 0
        self._collided[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.extras[env_ids] = 0


    def check_robot_collisions(self):
        # Check if the robot collided with an object
        # TODO: Parallelize
        for obst in self._obstacles:
            raw_readings = self._contact_sensor_interface.get_contact_sensor_raw_data(obst.prim_path + "/Contact_Sensor")
            if raw_readings.shape[0]:                
                for reading in raw_readings:
                    if "Tiago" in str(self._contact_sensor_interface.decode_body_name(reading["body1"])):
                        return True # Collision detected with some part of the robot
                    if "Tiago" in str(self._contact_sensor_interface.decode_body_name(reading["body0"])):
                        return True # Collision detected with some part of the robot
        for grasp_obj in self._grasp_objs:
            if grasp_obj == self._curr_grasp_obj: continue # Important. Exclude current target object for collision checking

            raw_readings = self._contact_sensor_interface.get_contact_sensor_raw_data(grasp_obj.prim_path + "/Contact_Sensor")
            if raw_readings.shape[0]:
                for reading in raw_readings:
                    if "Tiago" in str(self._contact_sensor_interface.decode_body_name(reading["body1"])):
                        return True # Collision detected with some part of the robot
                    if "Tiago" in str(self._contact_sensor_interface.decode_body_name(reading["body0"])):
                        return True # Collision detected with some part of the robot
        return False
    

    def calculate_metrics(self) -> None:
        # assuming data from obs buffer is available (get_observations() called before this function)

        if(self.check_robot_collisions()): # TODO: Parallelize
            # Collision detected. Give penalty and no other rewards
            self._collided[0] = 1
            self._is_success[0] = 0 # Success isn't considered in this case
            reward = torch.tensor(self._reward_collision, device=self._device)
        else:
            # Distance reward
            prev_goal_xy_dist = self._goals_xy_dist
            curr_goal_xy_dist = torch.linalg.norm(self.obs_buf[:,:2],dim=1)
            goal_xy_dist_reduction = (prev_goal_xy_dist - curr_goal_xy_dist).clone()
            reward = self._reward_dist_weight*goal_xy_dist_reduction
            # print(f"Goal Dist reward: {reward}")
            self._goals_xy_dist = curr_goal_xy_dist

            # IK fail reward (penalty)
            reward += self._reward_noIK*self._ik_fails

            # Success reward
            reward += self._reward_success*self._is_success
        # print(f"Total reward: {reward}")
        self.rew_buf[:] = reward
        self.extras[:] = self._is_success.clone() # Track success

    def is_done(self) -> None:
        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > np.pi / 2, 1, resets)
        # resets = torch.zeros(self._num_envs, dtype=int, device=self._device)
        
        # reset if success OR collided OR if reached max episode length
        resets = self._is_success.clone()
        resets = torch.where(self._collided.bool(), 1, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets