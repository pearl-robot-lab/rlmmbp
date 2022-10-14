# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import gc
import carb
import numpy as np
import cv2
import math
import copy
import pdb
import time

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import *
from mushroom_rl.utils.viewer import ImageViewer

from omni.isaac.python_app import OmniKitHelper

WIDTH_SCREEN = 1280 # 1600
HEIGHT_SCREEN = 720 # 900

ISAAC_PATH = os.environ.get('ISAAC_PATH')
USD_PATH = ISAAC_PATH + "/tiago_isaac/usd"
TIAGO_DUAL_HOME_USD_PATH = ISAAC_PATH + "/tiago_isaac/usd/tiago_dual_home.usd" # In home position and using kinect azure
TIAGO_DUAL_STAGE_PATH = "/tiago_dual"

CONFIG = {
    "experience": ISAAC_PATH + "/tiago_isaac/isaac_app_config/omni.isaac.sim.python_minimal.kit",
    "renderer": "RayTracedLighting",
    "display_options": 0,
    "headless": True,
    "window_width": 1920,
    "window_height": 1080,
    "width": WIDTH_SCREEN, # viewport
    "height": HEIGHT_SCREEN, # viewport
}

kit = OmniKitHelper(config=CONFIG)

# import omni requirements (Only after OmniKitHelper has started)
import omni
from pxr import UsdLux, Sdf, Gf, UsdPhysics, UsdGeom
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.syntheticdata import visualize
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.contact_sensor import _contact_sensor
from .utils import misc_utils
from .utils import math_utils
from .utils import control_utils
from .utils import ik_pykdl
from .utils.inv_reach_map import InvReachMap

class TiagoIsaacReaching(Environment):
    """
    Simple base placement environment for reaching
    """
    def __init__(self, world_xy_radius=3.0, action_xy_radius=1, action_xy_ang_lim=np.pi, action_yaw_lim=np.pi, action_arm='left', goal_pos_threshold=0.01, goal_ang_threshold=5*np.pi/180,
    reward_success=1., reward_dist_weight=0.1, reward_noIK=-0.05, reward_timeout=0.0, reward_collision=-0.15, gamma=0.95, horizon=10):
        """
        Constructor.
        Args:
            world_xy_radius (float, 3.0): radius of the world (environment)
            action_xy_radius (float, 1.0): radius around the robot for
             base placement action (in metres) (base xy action is in polar co-ordinates)
            action_xy_ang_lim (float, 1.0): polar angle limits for the x-y
             base placement action (in radians) (base xy action is in polar co-ordinates)
            action_yaw_lim (float, np.pi): yaw orientation (angular) action limits to be considered. (Default -pi to pi radians)
            action_arm (string, 'left' or 'right'): choose which arm of the robot to use for arm actions. TODO
            goal_pos_threshold (float, 0.05): distance threshold of the agent from the
                goal to consider it reached (metres);
            goal_ang_threshold (float, 15*pi/180): angular threshold of the agent from the
                goal to consider it reached (radians);
            reward_success (float, 1): reward obtained reaching goal state;
            reward_dist_weight (float, 0.1): weight for the reward for reducing distance to goal;
            reward_noIK (float, -0.05): reward obtained (negative) if no IK solution to goal is found;            
            reward_timeout (float, 0.0): reward obtained (negative) if time is up;
            reward_collision (float, -0.15): reward obtained (negative) if robot collides with an object
            gamma (float, 0.95): discount factor.
            horizon (int, 10): horizon of the problem;
        """
        # MDP parameters
        self._dt = 0.05 # timestep size for the environment = dt*sim_steps_per_dt (in seconds)
        self._sim_steps_per_dt = 1
        self._goal = np.array([0.7, 0.5, 0.49, 0., 0., 0.]) # 6D pose (metres, angles in euler extrinsic rpy (radians)) (Un-normalized)
        self._goal_state = copy.deepcopy(self._goal) # to track goal pose over time
        self._goal_within_reach = False # boolean to track if goal is within reach of Tiago (< action_space_radius)
        self._inv_reach_map = None # variable to store an inverse reachability map (if needed)
        self._world_xy_radius = world_xy_radius
        self._action_xy_radius = action_xy_radius # polar x-y co-ordinates
        self._action_xy_ang_lim = action_xy_ang_lim # polar x-y co-ordinates
        self._action_yaw_lim = action_yaw_lim
        self._action_arm = action_arm
        self._state = self.normalize_state(copy.deepcopy(self._goal_state))
        self._goal_pos_threshold = goal_pos_threshold
        self._goal_ang_threshold = goal_ang_threshold
        self._reward_success = reward_success
        self._reward_dist_weight = reward_dist_weight
        self._reward_noIK = reward_noIK
        self._reward_timeout = reward_timeout
        self._reward_collision = reward_collision
        self._horizon = horizon
        self._step_count = horizon
        self._done = True
        self._success = False

        # MDP properties
        action_space = Box(np.array([-1.,-1.,-1.,0.,0.]), np.array([1.,1.,1.,1.,1.]), shape=(5,))
        # action space is SE2 (x-y and orientation) plus 1 one-hot arm execution descision variable (will be discretized by the policy)
        
        observation_space = Box(-1., 1., shape=(6,)) # For now just 6D task pose
        # TODO: Add occupancy map to observation space
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        # Visualization
        self._viewer = ImageViewer([WIDTH_SCREEN, HEIGHT_SCREEN], self._dt)

        super().__init__(mdp_info)
        
        # Init Isaac Sim
        self.initTiagoIsaac()


    def reset(self, state=None):
        
        kit.stop() # This should rest all transforms to their init state (before 'play' was called)
        control_utils.set_torso_arm_left_joint_positions(self._stage, self._tiago_stage_path, np.zeros(8)) # Set joint targets to zero
        self._inv_reach_map = None # Reset the previously created inv reachability map
        gc.collect() # Save some CPU memory

        self._goal = self.spawn_objects() # Spawn props (objects) at randomized locations and return a goal object pose
        if (np.linalg.norm(self._goal[0:2]) <= self._action_xy_radius): self._goal_within_reach = True
        else: self._goal_within_reach = False

        kit.play() # play
        gt = self._sd_helper.get_groundtruth(["rgb"], self._viewport, verify_sensor_init=False)

        self._goal_state = copy.deepcopy(self._goal) # to track goal pose over time
        self._state = self.normalize_state(copy.deepcopy(self._goal_state))
        
        self._step_count = 0
        self._done = False
        self._success = False

        return self._state

    def step(self, action):
        if(self._done): # Already done. Step but with no change of state and no reward
            # Step robot with dt
            self.step_sim(self._sim_steps_per_dt, self._dt)
            return self._state, 0, self._done, {}
        
        # Keep actions within range (-1,1)
        self._bound(action, -1, 1)
        # assert ((action[3] in [0.0, 1.0]) and (action[4] in [0.0, 1.0])), "Descision variable must be discrete!"

        # Store previous goal state x and y (for calculating distance reward)
        prev_xy_goal_state = copy.deepcopy(self._goal_state[0:2])

        ## Take action, get next state
        # Scale actions and convert polar co-ordinates r-phi to x-y
        r_scaled = action[0]*self._action_xy_radius
        phi_scaled = action[1]*self._action_xy_ang_lim
        x_scaled = r_scaled*math.cos(phi_scaled)
        y_scaled = r_scaled*math.sin(phi_scaled)
        theta_scaled = action[2]*self._action_yaw_lim
        # Move robot
        # control_utils.setTiagoBasePose(self._dci, self._tiago_stage_path, x_scaled, y_scaled, np.degrees(theta_scaled)) # set base position and orientation in world frame
        control_utils.transformTiagoBase(self._dci, self._tiago_stage_path, x_scaled, y_scaled, np.degrees(theta_scaled)) # move base in robot frame
        # Move goal relative to robot (transform self._goal_state)
        actionTransform = Gf.Matrix4d().SetTranslateOnly(Gf.Vec3d(x_scaled,y_scaled,0.0)).SetRotateOnly(Gf.Rotation(Gf.Vec3d(0,0,1), np.degrees(theta_scaled)))
        goal_pose = Gf.Matrix4d().SetTranslateOnly(Gf.Vec3d(self._goal_state[0],self._goal_state[1],self._goal_state[2]))
        # Be careful of intrinsic and extrinsic rotations!
        goal_pose.SetRotateOnly(Gf.Rotation(Gf.Vec3d(1,0,0), np.degrees(self._goal_state[3]))*Gf.Rotation(Gf.Vec3d(0,1,0), np.degrees(self._goal_state[4]))*Gf.Rotation(Gf.Vec3d(0,0,1), np.degrees(self._goal_state[5])))
        goal_pose_transformed = goal_pose*actionTransform.GetInverse() # Gf uses row vector notation so order is reversed!
        self._goal_state[0:3] = np.array(goal_pose_transformed.ExtractTranslation())
        self._goal_state[3:6] = math_utils.quaternionToEulerAngles(goal_pose_transformed.ExtractRotation().GetQuaternion()) # Extrinsic Euler angles in radians
        # Step robot with dt
        self.step_sim(self._sim_steps_per_dt, self._dt)
        
        self._step_count += 1
        
        ## Get reward and check if done
        # Collision reward
        if(self.check_robot_collisions()):
            # Collision detected! Give penalty and end episode
            reward = self._reward_collision
            self._done = True
        else:
            # Distance reward
            new_goal_dist = np.linalg.norm(prev_xy_goal_state - [x_scaled, y_scaled])
            reward = self._reward_dist_weight*(np.linalg.norm(prev_xy_goal_state) - new_goal_dist)
            # Update 'goal_within_reach' boolean based on distance
            if (new_goal_dist <= self._action_xy_radius): self._goal_within_reach = True
            else: self._goal_within_reach = False
            # IK rewards
            if(action[3] > action[4]): # This is the arm decision variable
                # Solve IK to task pose
                # (Optional) Avoid backward IK plans:
                # if self._goal_state[0] < -0.3:
                #     reward += self._reward_noIK
                # else:
                    curr_joints = control_utils.dci_get_torso_arm_left_joint_positions(self._dci, self._tiago_stage_path)
                    reachable, ik_soln = ik_pykdl.solveIK_tiago_arm_left_torso(self._goal_state[0:3], self._goal_state[3:6], 
                    curr_joints, self._goal_pos_threshold, self._goal_ang_threshold, verbose=False)
                    if(reachable):
                        reward += self._reward_success
                        self._done = True
                        self._success = True
                        control_utils.set_torso_arm_left_joint_positions(self._stage, self._tiago_stage_path, ik_soln)                    
                    else:
                        reward += self._reward_noIK
            # Timeout reward
            if(self._step_count == self._horizon):
                reward += self._reward_timeout
                self._done = True

        self._state = self.normalize_state(copy.deepcopy(self._goal_state)) # Normalize before returning
        return self._state, reward, self._done, {}

    def step_sim(self, sim_steps_per_dt, dt):
        for frame in range(sim_steps_per_dt):
            kit.update(dt/sim_steps_per_dt) # Updating rendering and physics with a sim timestep of dt/sim_steps_per_dt

    def normalize_state(self, state):
        state[0:3] = state[0:3]/self._world_xy_radius
        state[3:6] = state[3:6]/np.pi
        return state

    def render(self):
        # DEBUG: If succesful, execute IK motion to see the reach
        if self._success:
            for frames in range(80):
                self.step_sim(self._sim_steps_per_dt, self._dt)

        # get gt rgb from main viewport
        gt = self._sd_helper.get_groundtruth(["rgb"],
                self._viewport,
                verify_sensor_init=False)
        
        self._viewer.display(gt["rgb"][:,:,0:3])
        time.sleep(self._dt) # Viz
    
    def get_render(self):
        # DEBUG: If succesful, execute IK motion to see the reach
        if self._success:
            for frames in range(30):
                self.step_sim(self._sim_steps_per_dt, self._dt)
        
        # return gt rgb from main viewport
        gt = self._sd_helper.get_groundtruth(["rgb"],
                self._viewport,
                verify_sensor_init=False)
        
        return gt["rgb"][:,:,0:3]

    def stop(self):
        #     self._viewer.close
        kit.stop()

    def shutdown(self):
        kit.stop()
        kit.shutdown()

    def initTiagoIsaac(self):
        # Preparing viewport
        self._viewport = omni.kit.viewport.get_default_viewport_window()
        self._viewport.set_camera_position("/OmniverseKit_Persp", 1.6*320, 1.6*-320, 1.6*410, True)
        self._viewport.set_camera_target("/OmniverseKit_Persp", 0, 0, 0, True)

        # Preparing stage
        self._stage = omni.usd.get_context().get_stage()
        self._dci = _dynamic_control.acquire_dynamic_control_interface()
        self._cs = _contact_sensor.acquire_contact_sensor_interface()

        # Optional: Wait so that stage starts loading
        # wait_load_stage(kit)

        # Spawn the Tiago++ 'home' robot (USD) (This will be the world frame)
        self._tiago_stage_path = TIAGO_DUAL_STAGE_PATH
        self._tiago_dual_usd_path = TIAGO_DUAL_HOME_USD_PATH
        self._tiagoPrim = self._stage.DefinePrim(self._tiago_stage_path, "Xform")
        self._tiagoPrim.GetReferences().AddReference(self._tiago_dual_usd_path)
        xform = UsdGeom.Xformable(self._tiagoPrim)
        transform = xform.AddTransformOp()

        ## Load the rest of the environment USDs
        # Physics scene
        scene = UsdPhysics.Scene.Define(self._stage, Sdf.Path("/physicsScene"))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(981.0)
        # Ground plane
        result, plane_path = omni.kit.commands.execute(
            "AddGroundPlaneCommand",
            stage=self._stage,
            planePath="/groundPlane",
            axis="Z",
            size=15000.0,
            position=Gf.Vec3f(0),
            color=Gf.Vec3f(0.06861, 0.108, 0.198),
        )
        # Lighting
        distantLight = UsdLux.DistantLight.Define(self._stage, Sdf.Path("/DistantLight"))
        distantLight.CreateIntensityAttr(500)        

        # Initialize sensors
        self._sd_helper = SyntheticDataHelper()
        gt = self._sd_helper.get_groundtruth(
            ["rgb"], self._viewport) # (Optional: Choose other viewports)

        # Optional: Wait so that stage starts loading
        # wait_load_stage(kit)

        # Optional: Simulate for two seconds to warm up sim and let everything settle
        # kit.play()
        # for frame in range(120):
        #     kit.update(1.0 / 60.0)

    def spawn_objects(self):
        ## Randomly generate the objects in the environment and return a goal object pose
        # TODO: Spawn n table-like objects and n YCB objects
        # TODO: Store and use dimensions of table-like objects
        # Exclude square at origin where the Tiago is
        tiago_xy_lim = 50 # cm
        world_xy_radius = self._world_xy_radius*100 # cm
        # polar co-ords
        random_r = np.random.uniform(tiago_xy_lim,world_xy_radius)
        rand_phi = np.random.uniform(-np.pi,np.pi)
        rand_x = random_r*math.cos(rand_phi)
        rand_y = random_r*math.sin(rand_phi)
        rand_z = np.random.uniform(10,140) # taking z_lim as 0.1m to 1.4m in the air
        rand_theta = np.random.uniform(-180,180) # Degrees

        ## Props
        # TODO: Remove existing props if they exist (ClearReferences())
        # Table-like: Chest
        # obj1_prim = self._stage.DefinePrim("/World/obj1", "Xform")
        # # For temporarily disabling a prim: obj1_prim.SetActive(False)
        # obj1_prim.GetReferences().AddReference(USD_PATH+"/Props/chest.usd")
        # trans = Gf.Vec3d(rand_x,rand_y,0.0)
        # rot = Gf.Rotation(Gf.Vec3d(0,0,1), rand_theta)
        # misc_utils.setPrimTransform(obj1_prim, trans, rot) # Note: This may sometimes 'transform' and not 'set' for certain rigid objects
        # obj1_pose = np.array([rand_x/100, rand_y/100, 0.0, 0., 0., np.radians(rand_theta)]) # 6D pose (metres, angles in euler (radians)) (Un-normalized)
        # # Setup contact sensor on prop for collision checking
        # UsdPhysics.CollisionAPI.Apply(obj1_prim) # Enable collision and contact sensing for prim
        # propts = _contact_sensor.SensorProperties()
        # propts.radius = -1  # Cover the entire body
        # propts.minThreshold = 0
        # propts.maxThreshold = 1000000000000
        # propts.sensorPeriod = self._dt
        # self._cs.add_sensor_on_body("/World/obj1", propts)

        # To be grasped/reached: Mustard Bottle
        rand_roll = np.random.uniform(-180,180) # Degrees        
        rand_yaw = np.random.uniform(-180,180) # Degrees
        # Note that pitch has half the range in euler angles
        if(rand_z<30):
            # use only upper hemishpere (grasps from top, downward facing)
            rand_pitch = np.random.uniform(0,90) # Positive Pitch
        elif(rand_z>120):
            # use only lower hemishpere 'intrinsic pitch' (grasps from bottom, upward facing)
            rand_pitch = np.random.uniform(0,-90) # Negative Pitch
        else:
            rand_pitch = np.random.uniform(-90,90)
        obj2_prim = self._stage.DefinePrim("/World/obj2", "Xform")
        obj2_prim.GetReferences().AddReference(USD_PATH+"/Props/YCB/Axis_Aligned/006_mustard_bottle.usd")
        trans = Gf.Vec3d(rand_x,rand_y,rand_z)
        rot = Gf.Rotation(Gf.Vec3d(1,0,0), rand_roll)*Gf.Rotation(Gf.Vec3d(0,1,0), rand_pitch)*Gf.Rotation(Gf.Vec3d(0,0,1), rand_yaw) # Extrinsic rotations in xyz
        misc_utils.setPrimTransform(obj2_prim, trans, rot)
        grasp_obj2_pose = np.array([rand_x/100, rand_y/100, rand_z/100, np.radians(rand_roll), np.radians(rand_pitch), np.radians(rand_yaw)]) # 6D pose (metres, angles in euler (radians)) (Un-normalized)
        
        # Return grasp/reach object pose.
        return grasp_obj2_pose

    def check_robot_collisions(self):
        # Check if the robot collided with a prop
        # TODO: Add more objects and check for collisions
        # raw_readings = self._cs.get_body_contact_raw_data("/World/obj1")
        # if raw_readings.shape[0]:
        #     for reading in raw_readings:
        #         if "tiago" in str(self._cs.decode_body_name(reading["body1"])):
        #             return True # Collision detected with some part of the robot
        return False

    def check_prior_condition(self): # condition to use a prior for biasing an RL agent
        return self._goal_within_reach

    def get_prior_action(self):
        ## Get action using the inverse reachability map
        if (self._inv_reach_map is None):
            # Create the map based on 6D goal pose
            self._inv_reach_map = InvReachMap(goal_pose=self._goal)

        curr_base_pose = control_utils.getTiagoBasePose(self._dci, self._tiago_stage_path)
        top_k = 0.075 # sampled base pose if from top k % of inv_reach_map
        vis_freq = True # whether to use Manipulability score or visitation frequency score
        base_pose = self._inv_reach_map.sample_base_pose(curr_base_pose=curr_base_pose, vis_freq=vis_freq, top_k=top_k,
                                max_distance=self._action_xy_radius, max_ang_distance=self._action_xy_ang_lim)
        # normalize base_pose and convert to polar co-ordinates
        r_norm = np.linalg.norm(base_pose[0:2])/self._action_xy_radius
        phi_norm = math.atan2(base_pose[1],base_pose[0])/self._action_xy_ang_lim
        theta_norm = base_pose[2]/self._action_yaw_lim
        action = np.hstack((r_norm, phi_norm, theta_norm, [1.0, 0.0])) # also use the arm
        # action = np.hstack((np.random.uniform(size=3)*2-1, [1.0, 0.0])) # Uniform action
        return action

    def get_prior_task_states(self, new_task_states, ik_task=False): # convert the state from a new task's state space to the prior task's state space                
        if(len(new_task_states.shape) < 2):
            # If received state is a single array, unsqueeze it
            new_task_states = np.array([new_task_states])
        prior_task_states = copy.deepcopy(new_task_states)
        weights = np.ones(shape=prior_task_states.shape[0]) # also return weighting for the states
        if(ik_task):
            # Prior task is simple IK 6D reaching in a 1m world radius
            prior_task_states[:,0:3] *= self._world_xy_radius # Re-scale xyz distances
            xy_distances = np.linalg.norm(prior_task_states[:,0:2], axis=1)

            # weights[np.linalg.norm(xy_distances > 1.0] = 0.0 # accept only states where xy distance is less than 1 metre
            
            # weights = np.clip(1/xy_distances, 0.0, 1.0) # weight as per 1/distance. Clip to maximum weight of 1.0
            
            weights = np.clip(1.0 - np.tanh(xy_distances-1.0), 0.0, 1.0) # Weights as per distance metric: 1-tanh. Clip to maximum weight of 1.0

            return weights, prior_task_states

        return weights, prior_task_states