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
import shapely
from shapely import geometry
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
TIAGO_DUAL_HOME_USD_PATH = ISAAC_PATH + "/tiago_isaac/usd/tiago_dual_home_kinect.usd" # In home position and using kinect azure
TIAGO_DUAL_STAGE_PATH = "/tiago_dual"

CONFIG = {
    "experience": ISAAC_PATH + "/tiago_isaac/isaac_app_config/omni.isaac.sim.python_minimal.kit",
    "renderer": "RayTracedLighting",
    "display_options": 0,
    "headless": False,
    "window_width": 1920,
    "window_height": 1080,
    "width": WIDTH_SCREEN, # viewport
    "height": HEIGHT_SCREEN, # viewport
}

kit = OmniKitHelper(config=CONFIG)

# import omni requirements (Only after OmniKitHelper has started)
import omni
from pxr import Usd, UsdLux, Sdf, Gf, UsdPhysics, UsdGeom
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
    def __init__(self, world_xy_radius=4.0, action_xy_radius=1, action_xy_ang_lim=np.pi, action_yaw_lim=np.pi, action_arm='left',
    num_obstacles=1, num_grasp_objs=4, goal_pos_threshold=0.01, goal_ang_threshold=15*np.pi/180, cubic_approx=True, terminate_on_collision=True,
    reward_success=1., reward_dist_weight=0.1, reward_noIK=-0.01, reward_timeout=0.0, reward_collision=-0.1, gamma=0.95, horizon=10):
        """
        Constructor.
        Args:
            world_xy_radius (float, 4.0): radius of the world (environment);
            action_xy_radius (float, 1.0): radius around the robot for
             base placement action (in metres) (base xy action is in polar co-ordinates);
            action_xy_ang_lim (float, 1.0): polar angle limits for the x-y
             base placement action (in radians) (base xy action is in polar co-ordinates);
            action_yaw_lim (float, np.pi): yaw orientation (angular) action limits to be considered. (Default -pi to pi radians);
            action_arm (string, 'left' or 'right'): choose which arm of the robot to use for arm actions; TODO
            num_obstacles (int, 1): Number of obstacles in the room environment
            num_grasp_objs (int, 4): Number of grasp objects in the room environment
                (all grasp objs except one will be used in obj state);
            goal_pos_threshold (float, 0.05): distance threshold of the agent from the
                goal to consider it reached (metres);
            goal_ang_threshold (float, 15*pi/180): angular threshold of the agent from the
                goal to consider it reached (radians);
            cubic_approx (bool, True): Whether to use a cubic approximation of the obstacles;
            terminate_on_collision (bool, False): Whether to terminate the episode when a collision is detected;
            reward_success (float, 1): reward obtained reaching goal state without collisions;
            reward_dist_weight (float, 0.1): weight for the reward for reducing distance to goal;
            reward_noIK (float, -0.05): reward obtained (negative) if no IK solution to goal is found;            
            reward_timeout (float, 0.0): reward obtained (negative) if time is up;
            reward_collision (float, -0.6): reward obtained (negative) if robot collides with an object
            gamma (float, 0.99): discount factor;
            horizon (int, 10): horizon of the problem.
        """
        # MDP parameters
        self._dt = 0.05 # timestep size for the environment = dt*sim_steps_per_dt (in seconds)
        self._sim_steps_per_dt = 1
        self._goal = np.array([0.7, 0.5, 0.49, 0., 0., 0.]) # 6D pose (metres, angles in euler extrinsic rpy (radians)) (Un-normalized)
        self._goal_state = copy.deepcopy(self._goal) # to track goal pose over time
        self._goal_within_reach = False # boolean to track if goal is within reach of Tiago (< action_space_radius)
        self._cubic_approx = cubic_approx
        self._terminate_on_collision = terminate_on_collision
        self._obstacle_names = ["mammut"]#, "godishus"] # Cubic models in usd format
        self._grasp_obj_names = ["004_sugar_box", "008_pudding_box", "010_potted_meat_can", "061_foam_brick"] # YCB models in usd format
        self._num_obstacles = min(num_obstacles,len(self._obstacle_names)) # number of obstacles to use in the room environment
        self._num_grasp_objs = min(num_grasp_objs,len(self._grasp_obj_names)) # number of grasp objects to use in the room environment
        self._obj_states = np.zeros(6*(self._num_obstacles+self._num_grasp_objs-1)) # All grasp objs except one will be used in obj state
        self._inv_reach_map = None # variable to store an inverse reachability map (if needed)
        self._world_xy_radius = world_xy_radius
        self._action_xy_radius = action_xy_radius # polar x-y co-ordinates
        self._action_xy_ang_lim = action_xy_ang_lim # polar x-y co-ordinates
        self._action_yaw_lim = action_yaw_lim
        self._action_arm = action_arm
        self._state = self.normalize_state(copy.deepcopy(np.hstack((self._goal_state, self._obj_states))))
        self._goal_pos_threshold = goal_pos_threshold
        self._goal_ang_threshold = goal_ang_threshold
        self._reward_success = reward_success
        self._reward_dist_weight = reward_dist_weight
        self._reward_noIK = reward_noIK
        self._reward_timeout = reward_timeout
        self._reward_collision = reward_collision
        self._horizon = horizon
        self._step_count = horizon
        self._in_collision = False # flag to signify if in collision
        self._tiago_tf = np.eye(3) # 2D transform of the tiago base. Starts at origin
        self._done = True
        self._success = False

        # MDP properties
        action_space = Box(np.array([-1.,-1.,-1.,0.,0.]), np.array([1.,1.,1.,1.,1.]), shape=(5,))
        # action space is SE2 (x-y and orientation) plus 1 one-hot arm execution descision variable (will be discretized by the policy)
        
        observation_space = Box(-1., 1., shape=(6+6*(self._num_obstacles+self._num_grasp_objs-1),)) # 6D task pose + 6D bbox for each obstacle in the room. All grasp objs except one will be used in obj state
        # TODO: Add occupancy map to observation space
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        # Visualization
        self._viewer = ImageViewer([WIDTH_SCREEN, HEIGHT_SCREEN], self._dt)

        super().__init__(mdp_info)
        
        # Init Isaac Sim
        self.initTiagoIsaac()


    def reset(self, state=None):
        
        kit.stop() # This should rest all transforms to their init state (before 'play' was called)
        self._tiago_tf = np.eye(3) # x,y and theta pose of the tiago base. Starts at origin
        control_utils.set_torso_arm_left_joint_positions(self._stage, self._tiago_stage_path, np.zeros(8)) # Set joint targets to zero
        self._inv_reach_map = None # Reset the previously created inv reachability map
        gc.collect() # Save some CPU memory
        
        # print(f"GOAL BEFORE: {self._goal}")
        self._goal, self._obj_states, self._collision_2D_boxes = self.spawn_objects() # Spawn props (objects) at randomized locations and return a goal object pose
        # print(f"GOAL AFTER: {self._goal}")
        if (np.linalg.norm(self._goal[0:2]) <= self._action_xy_radius): self._goal_within_reach = True
        else: self._goal_within_reach = False

        kit.play() # play
        # gt = self._sd_helper.get_groundtruth(["rgb"], self._viewport, verify_sensor_init=False)

        self._goal_state = copy.deepcopy(self._goal) # to track goal pose over time
        self._state = self.normalize_state(copy.deepcopy(np.hstack((self._goal_state, self._obj_states))))
        
        self._step_count = 0
        self._in_collision = False
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
        assert ((action[3] in [0.0, 1.0]) and (action[4] in [0.0, 1.0])), "Descision variable must be discrete!"

        # Store previous goal state x and y (for calculating distance reward)
        prev_xy_goal_state = copy.deepcopy(self._goal_state[0:2])

        ## Take action, get next state
        # Scale actions and convert polar co-ordinates r-phi to x-y
        r_scaled = action[0]*self._action_xy_radius
        phi_scaled = action[1]*self._action_xy_ang_lim
        x_scaled = r_scaled*math.cos(phi_scaled)
        y_scaled = r_scaled*math.sin(phi_scaled)
        theta_scaled = action[2]*self._action_yaw_lim
        # action transform (2D)
        action_tf = np.zeros((3,3))
        action_tf[:2,:2] = np.array([[np.cos(theta_scaled), -np.sin(theta_scaled)],[np.sin(theta_scaled), np.cos(theta_scaled)]])
        action_tf[:,-1] = np.array([x_scaled, y_scaled, 1.0])
        # Move robot
        if (self._in_collision):
            # Do an isaac-sim reset to avoid change in obstacle and robot poses due to the collision. TODO: This is a hack. Try to separate collisions and contact sensing
            kit.stop()
            kit.play()
            self._in_collision = False
        # Transform Tiago pose and set to new x,y, theta
        self._tiago_tf = self._tiago_tf @ action_tf
        control_utils.setTiagoBasePose(self._dci, self._tiago_stage_path, self._tiago_tf[0,-1], self._tiago_tf[1,-1], np.degrees(np.arctan2(self._tiago_tf[1,0],self._tiago_tf[0,0]))) # set base position and orientation in world frame
        # control_utils.transformTiagoBase(self._dci, self._tiago_stage_path, x_scaled, y_scaled, np.degrees(theta_scaled)) # move base in robot frame
        # Move goal relative to robot (transform self._goal_state)
        actionTransform = Gf.Matrix4d().SetTranslateOnly(Gf.Vec3d(x_scaled,y_scaled,0.0)).SetRotateOnly(Gf.Rotation(Gf.Vec3d(0,0,1), np.degrees(theta_scaled)))
        goal_pose = Gf.Matrix4d().SetTranslateOnly(Gf.Vec3d(self._goal_state[0],self._goal_state[1],self._goal_state[2]))
        # Be careful of intrinsic and extrinsic rotations!
        goal_pose.SetRotateOnly(Gf.Rotation(Gf.Vec3d(1,0,0), np.degrees(self._goal_state[3]))*Gf.Rotation(Gf.Vec3d(0,1,0), np.degrees(self._goal_state[4]))*Gf.Rotation(Gf.Vec3d(0,0,1), np.degrees(self._goal_state[5])))
        goal_pose_transformed = goal_pose*actionTransform.GetInverse() # Gf uses row vector notation so order is reversed!
        self._goal_state[0:3] = np.array(goal_pose_transformed.ExtractTranslation())
        self._goal_state[3:6] = math_utils.quaternionToEulerAngles(goal_pose_transformed.ExtractRotation().GetQuaternion()) # Extrinsic Euler angles in radians
        # Transform (2D) the objects' oriented bboxes:
        for obj_num in range(self._num_obstacles+self._num_grasp_objs-1):
            max_xy_vertex = np.array([[self._obj_states[6*obj_num],self._obj_states[6*obj_num+1],1.0]]).T
            min_xy_vertex = np.array([[self._obj_states[6*obj_num+2],self._obj_states[6*obj_num+3],1.0]]).T
            action_inv_tf = np.linalg.inv(action_tf)
            new_max_xy_vertex = (action_inv_tf @ max_xy_vertex)[0:2].T.squeeze()
            new_min_xy_vertex = (action_inv_tf @ min_xy_vertex)[0:2].T.squeeze()
            new_theta = self._obj_states[6*obj_num+5] - theta_scaled
            if (new_theta > np.pi): new_theta -= 2*np.pi
            elif (new_theta < -np.pi): new_theta += 2*np.pi
            # add Obbox to the state at the correct location
            self._obj_states[6*obj_num:6*obj_num+4] = np.array([new_max_xy_vertex[0],new_max_xy_vertex[1],
                                                               new_min_xy_vertex[0],new_min_xy_vertex[1]])
            self._obj_states[6*obj_num+5] = new_theta
            # (for obstacles only) compute collision 2D bboxes
            if(obj_num < self._num_obstacles):
                matrix = [action_inv_tf[0,0], action_inv_tf[0,1], action_inv_tf[1,0], action_inv_tf[1,1], action_inv_tf[0,2], action_inv_tf[1,2]]
                self._collision_2D_boxes[obj_num] = shapely.affinity.affine_transform(self._collision_2D_boxes[obj_num],matrix)

        # Step robot with dt
        self.step_sim(self._sim_steps_per_dt, self._dt)
        
        self._step_count += 1
        
        ## Get reward and check if done
        # Collision reward
        if(self.check_robot_collisions()):
            # Collision detected! Give penalty
            reward = self._reward_collision
            if (self._terminate_on_collision):
                self._done = True # terminate episode here
            else:
                self._in_collision = True # Set a flag to signify we are in collision but continue the episode
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
                # Avoid backward IK plans:
                if self._goal_state[0] < -0.3:
                    reward += self._reward_noIK
                else:
                    curr_joints = control_utils.dci_get_torso_arm_left_joint_positions(self._dci, self._tiago_stage_path)
                    reachable, ik_soln = ik_pykdl.solveIK_tiago_arm_left_torso(self._goal_state[0:3], self._goal_state[3:6], 
                    curr_joints, self._goal_pos_threshold, self._goal_ang_threshold, verbose=False)
                    if(reachable):
                        # Check for collision with other grasp objects
                        # First sim the IK solution
                        control_utils.set_torso_arm_left_joint_positions(self._stage, self._tiago_stage_path, ik_soln)
                        for frames in range(35):
                            self.step_sim(self._sim_steps_per_dt, self._dt)
                        
                        pdb.set_trace()

                        if(self.check_robot_collisions()):
                            # Collision detected! Give penalty
                            reward = self._reward_collision
                            if (self._terminate_on_collision):
                                self._done = True # terminate episode here
                            else:
                                self._in_collision = True # Set a flag to signify we are in collision but continue the episode
                        else:
                            reward += self._reward_success
                            self._done = True
                            self._success = True
                    else:
                        reward += self._reward_noIK
            # Timeout reward
            if(self._step_count == self._horizon):
                reward += self._reward_timeout
                self._done = True

        self._state = self.normalize_state(copy.deepcopy(np.hstack((self._goal_state, self._obj_states)))) # Normalize before returning

        return self._state, reward, self._done, {}

    def step_sim(self, sim_steps_per_dt, dt):
        for frame in range(sim_steps_per_dt):
            kit.update(dt/sim_steps_per_dt) # Updating rendering and physics with a sim timestep of dt/sim_steps_per_dt

    def normalize_state(self, state):
        state[0:3] = state[0:3]/self._world_xy_radius # Divide goal x,y,z by world radius
        state[3:6] = state[3:6]/np.pi # Divide goal euler angles by pi
        for obj_num in range(self._num_obstacles+self._num_grasp_objs-1):
            state[6+6*obj_num:6+6*obj_num+5] /= self._world_xy_radius # Divide obj bbox:x1,y1,x2,y2,z by world radius
            state[6+6*obj_num+5] /= np.pi # Divide bbox orientation by pi

        return state

    def render(self):
        # DEBUG: If succesful, execute IK motion to see the reach
        if self._success:
            pass
            # for frames in range(800): self.step_sim(self._sim_steps_per_dt, self._dt)

        # get gt rgb from main viewport
        gt = self._sd_helper.get_groundtruth(["rgb"],
                self._viewport,
                verify_sensor_init=False)
        
        self._viewer.display(gt["rgb"][:,:,0:3])
        time.sleep(self._dt) # Viz
    
    def get_render(self):
        # return dummy gt rgb from main viewport
        gt = self._sd_helper.get_groundtruth(["rgb"],
                self._viewport,
                verify_sensor_init=False)
        
        # DEBUG: If succesful, execute IK motion to see the reach
        if self._success:
            pass
            # for frames in range(35):
            #     self.step_sim(self._sim_steps_per_dt, self._dt)
        else:
            for frames in range(15):
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
        self._viewport.set_camera_position("/OmniverseKit_Persp", 3*320, 3*-320, 3*410, True)
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
        ## Randomly generate the objects in the environment and return a goal object pose. Ensure no overlaps, collisions!
        obj_states = np.zeros(6*(self._num_obstacles+self._num_grasp_objs-1)) # return 6D bbox for each obj

        # Exclude circle at origin where the Tiago is
        self._tiago_radius_cm = 35 # cm
        world_xy_radius = self._world_xy_radius*100 # cm
        
        # Use own collision detection for robot-obstacle collisions
        collision_2D_boxes = list()
        # (Optional): Use Isaac contact sensor
        contact_propts = _contact_sensor.SensorProperties() # Contact sensor properties for collision detection
        contact_propts.radius = -1  # Cover the entire body
        contact_propts.minThreshold = 0
        contact_propts.maxThreshold = 1000000000000
        contact_propts.sensorPeriod = self._dt

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_]) # bounding box calculator
        
        # To remove existing props if they exist try using ClearReferences()
        # For temporarily disabling a prim try obst_prim.SetActive(False)
        # for obst_num in range(self._num_obstacles): # Default is just one obstacle
        # Load object. Enable collision API and add contact sensor
        obst_prim = self._stage.DefinePrim("/World/obst0", "Xform")
        # obst_prim.GetReferences().ClearReferences() # Clear any old references (usd meshes) on prim
        # Random choice of obstacle
        obstacle_name = np.random.choice(self._obstacle_names)
        if (self._cubic_approx): # Optional: Use saved cubic approximation of the obstacles
            # Add wall to avoid robot going behind tables
            obst_prim.GetReferences().AddReference(USD_PATH+"/Props/Shapenet/"+obstacle_name+"/models/model_cubic_approx.usd") # Shapenet model in usd format
        else:
            # Add wall to avoid robot going behind tables
            obst_prim.GetReferences().AddReference(USD_PATH+"/Props/Shapenet/"+obstacle_name+"/models/model_normalized_white.usd") # Shapenet model in usd format         
        # UsdPhysics.CollisionAPI.Apply(obst_prim) # Enable collision and contact sensing for obstacle prim
        # self._cs.add_sensor_on_body("/World/obst0", contact_propts)

        rot = Gf.Rotation(Gf.Vec3d(1,0,0), 0)
        # Optional: Shapenet model may be downward facing. Rotate in X direction by 90 degrees
        rot = Gf.Rotation(Gf.Vec3d(1,0,0), 90)
        trans = Gf.Vec3d(0,0,0)
        misc_utils.setPrimTransform(obst_prim, trans, rot)
        # Compute bounding box of object
        bbox_cache.Clear()
        prim_bbox = bbox_cache.ComputeWorldBound(obst_prim)
        obst_xyz_size = prim_bbox.GetRange().GetSize()
        z_to_ground = -prim_bbox.GetRange().min[2]
        # Now find a valid world pose for this object:
        # Pre-computed: Valid pose is: positive/negative x (atleast 250 cm), orientation of min-max -+45 degrees
        # Set object to this pose (and place it on ground)
        # Also make sure you don't collide with tiago in the center
        obst_rand_x = np.random.choice([-1,1])*np.random.uniform(250,world_xy_radius) # taking max xy size margin from tiago
        obst_rand_y = np.random.uniform(-250,250)
        obst_rand_theta = -90 # degrees (TODO: test if this offset is necessary)
        if (obst_rand_x > 0.0 and obst_rand_y > 0.0):
            obst_rand_theta += np.random.uniform(0,45) # Degrees
        elif (obst_rand_x > 0.0 and obst_rand_y < 0.0):
            obst_rand_theta += np.random.uniform(-45,0) # Degrees
        elif (obst_rand_x < 0.0 and obst_rand_y > 0.0):
            obst_rand_theta += np.random.uniform(180,180-45) # Degrees
        elif (obst_rand_x < 0.0 and obst_rand_y < 0.0):
            obst_rand_theta += np.random.uniform(180-45,180) # Degrees
        misc_utils.setPrimTransform(obst_prim, Gf.Vec3d(obst_rand_x,obst_rand_y,z_to_ground), rot*Gf.Rotation(Gf.Vec3d(0,0,1), obst_rand_theta))
        
        # computed Oriented bbox based on the pose we have just set
        obst_rand_x /= 100 # metres
        obst_rand_y /= 100 # metres
        obst_rand_theta = np.radians(obst_rand_theta)
        bbox_tf = np.zeros((3,3))
        bbox_tf[:2,:2] = np.array([[np.cos(obst_rand_theta), -np.sin(obst_rand_theta)],[np.sin(obst_rand_theta), np.cos(obst_rand_theta)]])
        bbox_tf[:,-1] = np.array([obst_rand_x, obst_rand_y, 1.0])
        max_xy_vertex = np.array([[prim_bbox.GetRange().max[0]/100,prim_bbox.GetRange().max[1]/100,1.0]]).T # cm to metre
        min_xy_vertex = np.array([[prim_bbox.GetRange().min[0]/100,prim_bbox.GetRange().min[1]/100,1.0]]).T # cm to metre
        new_max_xy_vertex = (bbox_tf @ max_xy_vertex)[0:2].T.squeeze()
        new_min_xy_vertex = (bbox_tf @ min_xy_vertex)[0:2].T.squeeze()
        # add Obbox to the state at the correct location, remove unnecessary variables (Eg. we need only height and not z1,z2)
        obj_states[0:6] = np.array([new_max_xy_vertex[0],new_max_xy_vertex[1],
                                                            new_min_xy_vertex[0],new_min_xy_vertex[1],
                                                            obst_xyz_size[2]/100,obst_rand_theta]) # cm to metre
        # Compute collision 2D bboxes (using shapely)
        collision_box = geometry.box(min_xy_vertex[0], min_xy_vertex[1], max_xy_vertex[0], max_xy_vertex[1]) # minx, miny, maxx, maxy, ccw=True
        # transform based on the pose we have just set
        matrix = [bbox_tf[0,0], bbox_tf[0,1], bbox_tf[1,0], bbox_tf[1,1], bbox_tf[0,2], bbox_tf[1,2]]
        collision_box = shapely.affinity.affine_transform(collision_box,matrix)
        collision_2D_boxes.append(collision_box)
        obst_rand_x *= 100 # reset to cms
        obst_rand_y *= 100 # reset to cms
        
        ## Load the grasp objects
        # NOTE: In the state we will add the Oriented bboxes but for overlap detection between spawned objects we
        # use Axis Aligned bboxes since this ensures more separation and is simpler. TODO: Implement an Oriented bbox overlap test using shapely
        self._grasp_obj_AAbboxes = list() # list of axis aligned bounding boxes of the grasp objects
        grasp_objs_available = copy.deepcopy(self._grasp_obj_names)
        # Load the Goal object to be grasped/reached
        goal_obj_prim = self._stage.DefinePrim("/World/goal_obj", "Xform")
        goal_obj_prim.GetReferences().ClearReferences() # Clear any old references (usd meshes) on prim
        goal_obj_name = "008_pudding_box"#np.random.choice(self._grasp_obj_names)
        grasp_objs_available.remove(goal_obj_name) # remove it so we don't spawn this YCB object again
        goal_obj_prim.GetReferences().AddReference(USD_PATH+"/Props/YCB/Axis_Aligned/"+goal_obj_name+".usd") # YCB object in usd format
        # YCB models are downward facing. Rotate in X direction by -90 degrees
        trans = Gf.Vec3d(0,0,0)
        rot = Gf.Rotation(Gf.Vec3d(1,0,0), -90)
        misc_utils.setPrimTransform(goal_obj_prim, trans, rot) # Note: Test! This may sometimes 'transform' and not 'set' for certain rigid objects            
        # Now find a valid world pose for this object:
        # Compute bounding box of object
        bbox_cache.Clear()
        goal_obj_bbox = bbox_cache.ComputeWorldBound(goal_obj_prim)
        goal_obj_xzy_size = goal_obj_bbox.GetRange().GetSize() # For some reason YCB object bboxes are still as per prior rotation
        z_to_ground = - goal_obj_bbox.GetRange().min[1] # For YCB for some reason z is the [1] element
        # Pre-computed. Object at height of obstacle and spawned in a square around the min of x-y range of the obstacle (tabular)
        goal_rand_x = obst_rand_x + np.random.uniform(-min(obst_xyz_size[0:2])/2,min(obst_xyz_size[0:2])/2)
        goal_rand_y = obst_rand_y + np.random.uniform(-min(obst_xyz_size[0:2])/2,min(obst_xyz_size[0:2])/2)
        goal_rand_z = z_to_ground + obst_xyz_size[2] # Place on top of tabular obstacle
        # Set grasp obj to new random position and rotate further by a random yaw
        trans = Gf.Vec3d(goal_rand_x,goal_rand_y,goal_rand_z)
        goal_rand_yaw = 0.0#np.random.uniform(-178.0,178.0) # Degrees
        rot *= Gf.Rotation(Gf.Vec3d(0,0,1), goal_rand_yaw) # Extrinsic rotations in z
        misc_utils.setPrimTransform(goal_obj_prim, trans, rot)
        # Add new Axis Aligned bbox of this object to the list (to check for overlaps)
        # compute new AxisAligned bbox
        bbox_cache.Clear()
        goal_obj_bbox_new = bbox_cache.ComputeWorldBound(goal_obj_prim)
        self._grasp_obj_AAbboxes.append(goal_obj_bbox_new)
        # Generate only top hemisphere grasps: any roll, 0 to 90 (89) pitch, any yaw
        goal_rand_pitch = 88.0#np.random.uniform(0,88.0) # Degrees (avoid potential issues at 90 pitch)
        goal_rand_roll = 90.0#np.random.uniform(-178.0,178.0) # Degrees
        goal_rand_z += goal_obj_xzy_size[1]/2 + 20.0#np.random.uniform(-5,30) # Add (random) offset to object top (upto 30 cms)
        grasp_pose = np.array([goal_rand_x/100, goal_rand_y/100, goal_rand_z/100, np.radians(goal_rand_roll), np.radians(goal_rand_pitch), np.radians(goal_rand_yaw)]) # 6D pose (metres, angles in euler (radians)) (Un-normalized)
        
        # Load the other grasp objs with collisions enabled)
        for grasp_obj_num in range(self._num_grasp_objs-1):
            grasp_obj_name = grasp_objs_available[grasp_obj_num]
            # Load object. Enable collision API and add contact sensor
            grasp_obj_prim = self._stage.DefinePrim("/World/grasp_obj"+str(grasp_obj_num), "Xform")
            grasp_obj_prim.GetReferences().ClearReferences() # Clear any old references (usd meshes) on prim
            grasp_obj_prim.GetReferences().AddReference(USD_PATH+"/Props/YCB/Axis_Aligned/"+grasp_obj_name+".usd") # YCB object in usd format
            UsdPhysics.CollisionAPI.Apply(grasp_obj_prim) # Enable collision and contact sensing for prim
            self._cs.add_sensor_on_body("/World/grasp_obj"+str(grasp_obj_num), contact_propts)
            # YCB models are downward facing. Rotate in X direction by -90 degrees
            trans = Gf.Vec3d(0,0,0)
            rot = Gf.Rotation(Gf.Vec3d(1,0,0), -90)
            misc_utils.setPrimTransform(grasp_obj_prim, trans, rot) # Note: Test! This may sometimes 'transform' and not 'set' for certain rigid objects  
            
            # Compute bounding box of grasp object
            bbox_cache.Clear()
            grasp_obj_bbox = bbox_cache.ComputeWorldBound(grasp_obj_prim)
            grasp_obj_xzy_size = grasp_obj_bbox.GetRange().GetSize() # For some reason YCB object bboxes are still as per prior rotation
            z_to_ground = - grasp_obj_bbox.GetRange().min[1]
            # Now find a valid world pose for this object:
            while (1): # NOTE: Yes, this can go into an infinite loop if objects are many and big and obstacle (tabular) is too small
                # Spawn object at height of obstacle and in a square around the min of x-y range of the obstacle (tabular)
                obj_rand_x = obst_rand_x + np.random.uniform(-min(obst_xyz_size[0:2])/2,min(obst_xyz_size[0:2])/2)
                obj_rand_y = obst_rand_y + np.random.uniform(-min(obst_xyz_size[0:2])/2,min(obst_xyz_size[0:2])/2)
                obj_rand_z = z_to_ground + obst_xyz_size[2] # Place on top of tabular obstacle
                obj_rand_yaw = np.random.uniform(-180,180) # Degrees
                # Set to position but rotate only in yaw
                trans = Gf.Vec3d(obj_rand_x,obj_rand_y,obj_rand_z)
                misc_utils.setPrimTransform(grasp_obj_prim, trans, rot*Gf.Rotation(Gf.Vec3d(0,0,1), obj_rand_yaw))
                # compute new AxisAligned bbox
                bbox_cache.Clear()
                grasp_obj_bbox_new = bbox_cache.ComputeWorldBound(grasp_obj_prim)
                # Check for overlap with existing objects
                overlap = False
                for other_aabbox in self._grasp_obj_AAbboxes: # loop over existing AAbboxes
                    intersec = Gf.Range3d.GetIntersection(grasp_obj_bbox_new.ComputeAlignedRange(), other_aabbox.ComputeAlignedRange())
                    if (not intersec.IsEmpty()):
                        overlap = True # Failed. Try another pose
                        break
                if (overlap):
                    continue # Failed. Try another pose
                else:
                    # Success. Add this valid AAbbox to the list
                    self._grasp_obj_AAbboxes.append(grasp_obj_bbox_new)
                    break
            # computed Oriented bbox based on the pose we have just set
            obj_rand_x /= 100 # metres
            obj_rand_y /= 100 # metres
            obj_rand_yaw = np.radians(obj_rand_yaw)
            bbox_tf = np.zeros((3,3))
            bbox_tf[:2,:2] = np.array([[np.cos(obj_rand_yaw), -np.sin(obj_rand_yaw)],[np.sin(obj_rand_yaw), np.cos(obj_rand_yaw)]])
            bbox_tf[:,-1] = np.array([obj_rand_x, obj_rand_y, 1.0])
            max_xy_vertex = np.array([[grasp_obj_bbox.GetRange().max[0]/100,grasp_obj_bbox.GetRange().max[1]/100,1.0]]).T # cm to metre
            min_xy_vertex = np.array([[grasp_obj_bbox.GetRange().min[0]/100,grasp_obj_bbox.GetRange().min[1]/100,1.0]]).T # cm to metre
            new_max_xy_vertex = (bbox_tf @ max_xy_vertex)[0:2].T.squeeze()
            new_min_xy_vertex = (bbox_tf @ min_xy_vertex)[0:2].T.squeeze()
            # add Obbox to the state at the correct location, remove unnecessary variables (Eg. we need only height and not z1,z2)
            obj_states[6*(self._num_obstacles+grasp_obj_num):6*(self._num_obstacles+grasp_obj_num+1)] = np.array([new_max_xy_vertex[0],new_max_xy_vertex[1],
                                                               new_min_xy_vertex[0],new_min_xy_vertex[1],
                                                               grasp_obj_xzy_size[1]/100,obj_rand_yaw]) # cm to metre

        # Return grasp/reach object pose.
        return grasp_pose, obj_states, collision_2D_boxes

    def check_robot_collisions(self):
        # Check if the robot collided with a prop
        
        # Use simple collision detection of base with obstacles (using shapely)
        tiago_circle = shapely.geometry.Point(0, 0).buffer(self._tiago_radius_cm/100)
        for idx, collision_box in enumerate(self._collision_2D_boxes):
            if collision_box.intersects(tiago_circle):
                return True

        # Use Isaac contact sensing (sometimes unreliable)
        for obst_num in range(self._num_obstacles): # Check for all obstacles
            raw_readings = self._cs.get_body_contact_raw_data("/World/obst"+str(obst_num))
            if raw_readings.shape[0]:
                for reading in raw_readings:
                    if "tiago" in str(self._cs.decode_body_name(reading["body1"])):
                        return True # Collision detected with some part of the robot
                    if "tiago" in str(self._cs.decode_body_name(reading["body0"])):
                        return True # Collision detected with some part of the robot

        for grasp_obj_num in range(self._num_grasp_objs): # Check for all grasp objs
            raw_readings = self._cs.get_body_contact_raw_data("/World/grasp_obj"+str(grasp_obj_num))
            if raw_readings.shape[0]:
                for reading in raw_readings:
                    if "tiago" in str(self._cs.decode_body_name(reading["body1"])):
                        return True # Collision detected with some part of the robot
                    if "tiago" in str(self._cs.decode_body_name(reading["body0"])):
                        return True # Collision detected with some part of the robot

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
        weights = np.ones(shape=new_task_states.shape[0]) # also return weighting for the states
        # Remove excess state variables as prior tasks are free space reaching tasks
        prior_task_states = copy.deepcopy(new_task_states[:,0:6]) # only using the 6D goal state

        # if(ik_task):
        #     # Prior task is simple IK 6D reaching in a 1m world radius
        #     prior_task_states[:,0:3] *= self._world_xy_radius # Re-scale xyz distances
            # xy_distances = np.linalg.norm(prior_task_states[:,0:2], axis=1)

            # weights[np.linalg.norm(xy_distances > 1.0] = 0.0 # accept only states where xy distance is less than 1 metre
            
            # weights = np.clip(1/xy_distances, 0.0, 1.0) # weight as per 1/distance. Clip to maximum weight of 1.0
            
            # weights = np.clip(1.0 - np.tanh(xy_distances-1.0), 0.0, 1.0) # Weights as per distance metric: 1-tanh. Clip to maximum weight of 1.0
        # else:
        # prior task is a 5m free reaching task
        prior_task_states[:,0:3] *= self._world_xy_radius # Re-scale xyz distances
        # convert to distances for prior 5m reaching task
        prior_task_states[:,0:3] /= 5.0

        return weights, prior_task_states