# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from omni.isaac.kit import SimulationApp
import os
import argparse
import pdb

# Set up command line arguments
parser = argparse.ArgumentParser("Tiago isaac launch")
parser.add_argument("--headless", default=False, action="store_true", help="Run sim headless")
parser.add_argument("--sim_dt", type=float, default=1.0/60.0, help="Step simulation with timestep dt")
parser.add_argument("--sim_realtime", default=False, action="store_true", help="Step simulation in realtime i.e. with timestep dt as small as possible")
args, unknown = parser.parse_known_args()

RENDER_WIDTH = 1280 # 1600
RENDER_HEIGHT = 720 # 900
scene_objs_usd_path = None
scene_objs_usd_path = '/home/sjauhri/Downloads/viz_assests/room_floorplan.usd'

ISAAC_PATH = os.environ.get('ISAAC_PATH')
TIAGO_DUAL_URDF_PATH = ISAAC_PATH + "/rlmmbp/learned_robot_placement/urdf/tiago_dual_holobase.urdf"
TIAGO_DUAL_USD_PATH = ISAAC_PATH + "/rlmmbp/learned_robot_placement/usd/tiago_dual_holobase/tiago_dual_holobase_zed_w_object.usd"
TIAGO_DUAL_STAGE_PATH = "/tiago_dual"

simulation_app = SimulationApp({#"experience": sim_app_cfg_path,
                                              "headless": args.headless,
                                              "window_width": 1920,
                                              "window_height": 1080,
                                              "width": RENDER_WIDTH,
                                              "height": RENDER_HEIGHT})

# import omni requirements (Only after App has started)
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.materials.omni_glass import OmniGlass
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.utils.stage import add_reference_to_stage

import omni
import carb
import numpy as np
import torch

# import tiago dual robot articulation
from learned_robot_placement.robots.articulations.tiago_dual_holo import TiagoDualHolo
# import sensors that you need
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor._range_sensor import acquire_lidar_sensor_interface
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
# Viz
from learned_robot_placement.utils.visualisation_utils import Visualizer
from omni.isaac.core.utils.prims import get_prim_parent, get_prim_at_path, set_prim_property, get_prim_property
# from omni.isaac.core.prims.rigid_prim import RigidPrim

class SimpleTiagoHandler:

    def __init__(self, robot, move_group='arm_right', use_torso=True, device='cpu'):
        
        self.robot = robot
        self.move_group = move_group
        self.use_torso = use_torso
        self.device = device
        
        # joint names
        self.base_joint_names = ["X",
                                "Y",
                                "R"]
        self.torso_joint_name = ["torso_lift_joint"]
        self.arm_left_names = []
        self.arm_right_names = []
        for i in range(7):
            self.arm_left_names.append(f"arm_left_{i+1}_joint")
            self.arm_right_names.append(f"arm_right_{i+1}_joint")
        
        # Future: Use end-effector link names and get their poses and velocities from Isaac
        self.ee_left_prim =  ["gripper_left_grasping_frame"]
        self.ee_right_prim = ["gripper_right_grasping_frame"]
        self.gripper_left_names = ["gripper_left_left_finger_joint","gripper_left_right_finger_joint"]
        self.gripper_right_names = ["gripper_right_left_finger_joint","gripper_right_right_finger_joint"]
        self.arm_left_start = torch.tensor([0.0, 1.5708, 1.5708,
                                            1.5708, 1.5708, -1.5708, 1.5708], device=device)
        self.arm_right_start = torch.tensor([0.0, 1.5708, 1.5708,
                                            1.5708, 1.5708, -1.0708, 1.5708], device=device)
        self.gripper_left_start = torch.tensor([0.045, 0.045], device=device) # Opened gripper by default
        self.gripper_right_start = torch.tensor([0.045, 0.045], device=device) # Opened gripper by default
        self.torso_fixed_state = torch.tensor([0.25], device=device)
        self.base_dof_idxs = []
        self.torso_dof_idx = []
        self.arm_left_dof_idxs = []
        self.arm_right_dof_idxs = []
        self.gripper_left_dof_idxs = []
        self.gripper_right_dof_idxs = []
        self.upper_body_dof_idxs = []
        self.combined_dof_idxs = []
        [self.base_dof_idxs.append(robot.get_dof_index(name)) for name in self.base_joint_names]
        [self.torso_dof_idx.append(robot.get_dof_index(name)) for name in self.torso_joint_name]
        [self.arm_left_dof_idxs.append(robot.get_dof_index(name)) for name in self.arm_left_names]
        [self.arm_right_dof_idxs.append(robot.get_dof_index(name)) for name in self.arm_right_names]
        [self.gripper_left_dof_idxs.append(robot.get_dof_index(name)) for name in self.gripper_left_names]
        [self.gripper_right_dof_idxs.append(robot.get_dof_index(name)) for name in self.gripper_right_names]
        if use_torso:
            self.upper_body_dof_idxs += self.torso_dof_idx
        if move_group == "arm_left":
            self.upper_body_dof_idxs += self.arm_left_dof_idxs
        elif move_group == "arm_right":
            self.upper_body_dof_idxs += self.arm_right_dof_idxs
        elif move_group == "both_arms":
            self.upper_body_dof_idxs += self.arm_left_dof_idxs + self.arm_right_dof_idxs
        else:
            raise ValueError('move_group not defined')
        self.combined_dof_idxs = self.base_dof_idxs + self.upper_body_dof_idxs

        joint_states = robot.get_joints_default_state()
        jt_pos = joint_states.positions
        jt_pos[self.torso_dof_idx] = self.torso_fixed_state
        jt_pos[self.arm_left_dof_idxs] = self.arm_left_start
        jt_pos[self.arm_right_dof_idxs] = self.arm_right_start
        jt_pos[self.gripper_left_dof_idxs] = self.gripper_left_start
        jt_pos[self.gripper_right_dof_idxs] = self.gripper_right_start
        
        robot.set_joints_default_state(positions=jt_pos)

    def set_gripper_right_effort(self, gripper_effort):
        # efforts
        # gripper_effort = gripper_effort.repeat((1,3))
        # gripper_effort[:, -1] *= -1
        # gripper_effort[:, -2] *= -1
        self.robot.set_joint_efforts(efforts=gripper_effort, joint_indices=self.gripper_right_dof_idxs)

    def set_gripper_left_effort(self, gripper_effort):
        # efforts
        # gripper_effort = gripper_effort.repeat((1,3))
        # gripper_effort[:, -1] *= -1
        # gripper_effort[:, -2] *= -1
        self.robot.set_joint_efforts(efforts=gripper_effort, joint_indices=self.gripper_left_dof_idxs)
    
    def set_gripper_right_positions(self, gripper_position):
        # targets
        self.robot._articulation_view.set_joint_position_targets(positions=gripper_position, joint_indices=self.gripper_right_dof_idxs)
        # direct positions
        # self.robot.set_joint_positions(positions=gripper_position, joint_indices=self.gripper_right_dof_idxs)

    def set_gripper_left_positions(self, gripper_position):
        # targets
        self.robot._articulation_view.set_joint_position_targets(positions=gripper_position, joint_indices=self.gripper_left_dof_idxs)
        # direct positions
        # self.robot.set_joint_positions(positions=gripper_position, joint_indices=self.gripper_left_dof_idxs)


def generate_camera(camera_path="/World/TiagoDualHolo/zed_camera_center/ZedCamera", intrinsics=None):
    camera = Camera(prim_path=camera_path,
                    name=f"zed_camera",
                    frequency=120)

    if camera.is_valid():
        camera.initialize()
        camera.add_pointcloud_to_frame()
        camera.add_distance_to_image_plane_to_frame()
        camera.add_semantic_segmentation_to_frame()
        camera.add_instance_id_segmentation_to_frame()
        camera.add_instance_segmentation_to_frame()
        camera.set_clipping_range(0.05, 50)
        if intrinsics is not None:
            raise NotImplementedError("Need to copy over set_intrinsics")
            # set_intrinsics(intrinsics)
    else:
        RuntimeError("Camera Path is not valid")
    
    return camera

def generate_lidar(path="/World"+"/TiagoDualHolo"+"/base_footprint"+"/Lidar", translation=Gf.Vec3f(0, 0, 0.15), orientation=Gf.Quatf(0, 0, 0, 1)):
    result, lidar = omni.kit.commands.execute(
        "RangeSensorCreateLidar",
        path=path,
        parent=None,
        min_range=0.3,
        max_range=100.0,
        draw_points=False,
        draw_lines=False,
        horizontal_fov=360.0,
        vertical_fov=30.0,
        horizontal_resolution=0.4,
        vertical_resolution=4.0,
        rotation_rate=0.0,
        high_lod=False,
        yaw_offset=0.0,
        enable_semantics=False,
    )

    lidar_prim = lidar.GetPrim()

    if "xformOp:translate" not in lidar_prim.GetPropertyNames():
        UsdGeom.Xformable(lidar_prim).AddTranslateOp()
    if "xformOp:orient" not in lidar_prim.GetPropertyNames():
        UsdGeom.Xformable(lidar_prim).AddOrientOp()

    lidar_prim.GetAttribute("xformOp:translate").Set(translation)
    lidar_prim.GetAttribute("xformOp:orient").Set(orientation)

    return result, lidar



if __name__ == "__main__":
    
    ## Initialize extensions and UI elements
    # Extensions
    # ext_manager = omni.kit.app.get_app().get_extension_manager()
    
    simulation_world = World(stage_units_in_meters=1.0)
    set_camera_view(eye=np.array([-4, 4, 4.0222]), target=np.array([0.0, 0.0, 0.0]))

    # Step our simulation to ensure everything initialized
    simulation_world.step()

    # Add a distant light
    create_prim("/DistantLight", "DistantLight", attributes={"intensity": 500})

    # Add a ground collision plane
    # simulation_world.scene.add_ground_plane(size=1000, z_position=-0.5, color=np.array([1, 1, 1]))

    # Add scene objects
    if scene_objs_usd_path is not None:
        add_reference_to_stage(scene_objs_usd_path, '/World/Scene')

    # Create robot & add it to the scene
    tiago = TiagoDualHolo(prim_path="/World" + "/TiagoDualHolo", name="TiagoDualHolo", usd_path=TIAGO_DUAL_USD_PATH,
                translation=torch.tensor([0.821, 0, 0.1425]), orientation=torch.tensor([-0.25, 0, 0, 0.96]))
    simulation_world.scene.add(tiago)
    simulation_world.reset() # important to ensure everything is initialized
    tiago_handler = SimpleTiagoHandler(tiago, use_torso=True, device='cpu') # control the robot
    simulation_world.reset() # important to ensure everything is initialized

    # Add sensors
    # cam = generate_camera(camera_path="/World/TiagoDualHolo/zed_camera_center/ZedCamera", intrinsics=None)
    lidar_path = "/World"+"/TiagoDualHolo"+"/base_footprint"+"/Lidar"
    result, lidar_prim = generate_lidar(path=lidar_path)
    simulation_world.reset() # important to ensure everything is initialized
    lidar_sensor_interface = acquire_lidar_sensor_interface()
    if lidar_sensor_interface.is_lidar_sensor(lidar_path):
        print("lidar sensor is valid")

    # Optional: Grasp an object
    for i in range(30): simulation_world.step(render=True)
    # import pdb; pdb.set_trace()
    # for i in range(1500): simulation_world.render()
    obj_prim_path = "/World/Scene/Objects/Box_1"
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(get_prim_at_path('/World/TiagoDualHolo/arm_right_7_link/_04_sugar_box'))
    rigid_api.CreateRigidBodyEnabledAttr(True)
    tiago_handler.set_gripper_right_positions(torch.Tensor([[0.015, 0.015]]))
    # tiago_handler.set_gripper_right_effort(torch.Tensor([-50.0,-50.0]))
    # arm_right_7_prim_path = '/World/TiagoDualHolo/arm_right_7_link'
    arm_right_7_prim_path = '/World/TiagoDualHolo/gripper_right_right_finger_link'
    arm_right_7_prim = get_prim_at_path(arm_right_7_prim_path)
    # arm_right_7_prim = RigidPrim(tiago_handler.robot._articulation_view._dof_paths[0][tiago_handler.arm_right_dof_idxs[-1]])
    viz = Visualizer()
    
    # Run sim app
    while simulation_app.is_running():
        simulation_world.step(render=True)
        
        # Visualize sensor data (TODO)
        # get force sensor readings on arm_right_7_link
        sensor_readings = tiago_handler.robot._articulation_view._physics_view.get_force_sensor_forces()
        xyz_readings = sensor_readings[0][0][:3]
        scaled_readings = xyz_readings * 0.08
        maxed_scaled_readings = np.clip(scaled_readings, -0.3, 0.3)
        # swap y and z
        maxed_scaled_readings[[1,2]] = maxed_scaled_readings[[2,1]]
        # reduce y by half
        maxed_scaled_readings[1] /= 2
        # flip direction of z
        maxed_scaled_readings[[2]] *= -1
        print("sensor readings: ", sensor_readings, "scaled readings: ", maxed_scaled_readings)

        # pos_quat = arm_right_7_prim.get_world_pose()
        tf_mat = omni.usd.get_world_transform_matrix(arm_right_7_prim)
        pos_gf, quat_gf = tf_mat.ExtractTranslation(), tf_mat.ExtractRotation().GetQuaternion()
        pos = [pos_gf[0], pos_gf[1], pos_gf[2]]
        # quat = [quat_gf.real, quat_gf.imaginary[0], quat_gf.imaginary[1], quat_gf.imaginary[2]]
        quat = [1, 0, 0, 0] # TEMP
        viz.draw_frame_pos_quat((pos,quat), axis_length=maxed_scaled_readings)