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
TIAGO_DUAL_USD_PATH = ISAAC_PATH + "/rlmmbp/learned_robot_placement/usd/tiago_dual_holobase/tiago_dual_holobase.usd"
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

def setup_robot(robot, move_group='arm_right', use_torso=True, device='cpu'):
    # joint names
    base_joint_names = ["X",
                              "Y",
                              "R"]
    torso_joint_name = ["torso_lift_joint"]
    arm_left_names = []
    arm_right_names = []
    for i in range(7):
        arm_left_names.append(f"arm_left_{i+1}_joint")
        arm_right_names.append(f"arm_right_{i+1}_joint")
    
    # Future: Use end-effector link names and get their poses and velocities from Isaac
    ee_left_prim =  ["gripper_left_grasping_frame"]
    ee_right_prim = ["gripper_right_grasping_frame"]
    gripper_left_names = ["gripper_left_left_finger_joint","gripper_left_right_finger_joint"]
    gripper_right_names = ["gripper_right_left_finger_joint","gripper_right_right_finger_joint"]
    arm_left_start = torch.tensor([0.0, 1.5708, 1.5708,
                                        1.5708, 1.5708, -1.5708, 1.5708], device=device)
    arm_right_start = torch.tensor([0.0, 1.5708, 1.5708,
                                        1.5708, 1.5708, -1.5708, 1.5708], device=device)
    gripper_left_start = torch.tensor([0.045, 0.045], device=device) # Opened gripper by default
    gripper_right_start = torch.tensor([0.045, 0.045], device=device) # Opened gripper by default
    torso_fixed_state = torch.tensor([0.25], device=device)
    base_dof_idxs = []
    torso_dof_idx = []
    arm_left_dof_idxs = []
    arm_right_dof_idxs = []
    gripper_left_dof_idxs = []
    gripper_right_dof_idxs = []
    upper_body_dof_idxs = []
    combined_dof_idxs = []
    [base_dof_idxs.append(robot.get_dof_index(name)) for name in base_joint_names]
    [torso_dof_idx.append(robot.get_dof_index(name)) for name in torso_joint_name]
    [arm_left_dof_idxs.append(robot.get_dof_index(name)) for name in arm_left_names]
    [arm_right_dof_idxs.append(robot.get_dof_index(name)) for name in arm_right_names]
    [gripper_left_dof_idxs.append(robot.get_dof_index(name)) for name in gripper_left_names]
    [gripper_right_dof_idxs.append(robot.get_dof_index(name)) for name in gripper_right_names]
    upper_body_dof_idxs = []
    if use_torso:
        upper_body_dof_idxs += torso_dof_idx
    if move_group == "arm_left":
        upper_body_dof_idxs += arm_left_dof_idxs
    elif move_group == "arm_right":
        upper_body_dof_idxs += arm_right_dof_idxs
    elif move_group == "both_arms":
        upper_body_dof_idxs += arm_left_dof_idxs + arm_right_dof_idxs
    else:
        raise ValueError('move_group not defined')
    combined_dof_idxs = base_dof_idxs + upper_body_dof_idxs

    joint_states = robot.get_joints_default_state()
    jt_pos = joint_states.positions
    jt_pos[torso_dof_idx] = torso_fixed_state
    jt_pos[arm_left_dof_idxs] = arm_left_start
    jt_pos[arm_right_dof_idxs] = arm_right_start
    jt_pos[gripper_left_dof_idxs] = gripper_left_start
    jt_pos[gripper_right_dof_idxs] = gripper_right_start
    
    robot.set_joints_default_state(positions=jt_pos)



if __name__ == "__main__":
    
    ## Initialize extensions and UI elements
    # Extensions
    # ext_manager = omni.kit.app.get_app().get_extension_manager()
    
    simulation_world = World(stage_units_in_meters=1.0)
    set_camera_view(eye=np.array([-0.9025, 2.1035, 1.0222]), target=np.array([0.6039, 0.30, 0.0950]))

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
    tiago = TiagoDualHolo(prim_path="/World" + "/TiagoDualHolo", name="TiagoDualHolo",
                translation=torch.tensor([0.821, 0, 0.1425]), orientation=torch.tensor([-0.25, 0, 0, 0.96]))
    simulation_world.scene.add(tiago)
    simulation_world.reset() # important to ensure everything is initialized
    setup_robot(tiago, use_torso=True, device='cpu')
    simulation_world.reset() # important to ensure everything is initialized

    # Run sim app
    while simulation_app.is_running():
        simulation_world.step(render=True)
        
        # Visualize sensor data (TODO)

    # _usd_context = omni.usd.get_context()
    # _stage = _usd_context.get_stage()


    # # Wait so that stage starts loading
    # wait_load_stage()

    # # Spawn the Tiago++ robot (USD) (This will be the world frame)
    # tiago_stage_path = TIAGO_DUAL_STAGE_PATH
    # tiago_dual_usd_path = TIAGO_DUAL_USD_PATH
    # tiagoPrim = _stage.DefinePrim(tiago_stage_path, "Xform")
    # tiagoPrim.GetReferences().AddReference(tiago_dual_usd_path)
    # # No translation  or rotation because we want this as world frame
    # # trans = Gf.Vec3d(0.0,0.0,0.0)
    # # rot = Gf.Rotation(Gf.Vec3d(0,0,1), 0)
    # # transformPrim(tiagoPrim, trans, rot)

    # ## Load the rest of the environment USDs
    # # Physics scene
    # scene = UsdPhysics.Scene.Define(_stage, Sdf.Path("/physicsScene"))
    # scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    # scene.CreateGravityMagnitudeAttr().Set(981.0)
    # # Ground plane
    # result, plane_path = omni.kit.commands.execute(
    #     "AddGroundPlaneCommand",
    #     stage=_stage,
    #     planePath="/groundPlane",
    #     axis="Z",
    #     size=1500.0,
    #     position=Gf.Vec3f(0),
    #     color=Gf.Vec3f(0.06861, 0.108, 0.198),
    # )
    # # Lighting
    # distantLight = UsdLux.DistantLight.Define(_stage, Sdf.Path("/DistantLight"))
    # distantLight.CreateIntensityAttr(500)


    # # Wait so that stage starts loading
    # wait_load_stage()

    # kit.play()
    # kit.update(1.0 / 60.0)

    # # Simulate for one second to warm up sim and let everything settle
    # for frame in range(60):
    #     kit.update(1.0 / 60.0)
    # # Initialize sensors
    # sd_helper = SyntheticDataHelper()
    # gt = sd_helper.get_groundtruth(
    #     ["rgb","depth"],
    #     _viewport, # (Optional: Choose other viewports)
    #     )
    
    # # If rendering separately, use render_iter count and render in editor step callback
    # if(args.render):
    #     render_iter = 0
        
    #     # Create callback to editor step
    #     def editor_update(e: carb.events.IEvent):
    #         # Render viewport after every render_interval
    #         global render_iter
    #         if(render_iter % args.render_interval == 0):
    #             render_iter = 0
    #             gt = sd_helper.get_groundtruth(
    #                 ["rgb"],
    #                 _viewport, # (Optional: Choose other viewports)
    #                 verify_sensor_init=False
    #                 )
    #             # RGB
    #             rgb_img = gt["rgb"]
    #             bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    #             if(args.render_depth):
    #                 gt_depth = sd_helper.get_groundtruth(
    #                     ["depth"],
    #                     _viewport, # (Optional: Choose other viewports)
    #                     verify_sensor_init=False
    #                     )
    #                 # Depth
    #                 depth_data = np.clip(gt_depth["depth"], 0, 255)
    #                 depth_img = visualize.colorize_depth(depth_data.squeeze())
    #                 depth_img_3d = depth_img[:,:,0:3].astype(np.uint8)

    #                 # concat
    #                 bgr_img = np.vstack((bgr_img, depth_img_3d))

    #             # Visualize
    #             cv2.imshow('Viewport: Perspective', bgr_img)
    #             cv2.waitKey(1)
    #         render_iter+=1
        
    #     update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(editor_update)

    # kit.play()
    
    # while kit.app.is_running():

    #     if(args.sim_realtime):
    #         # Run in realtime:
    #         kit.update()
    #     else:
    #         # Run with a fixed step size:
    #         # import time
    #         # curr_time = time.time()
    #         kit.update(args.sim_dt) # Updating rendering and physics with a sim timestep of dt
    #         # dt = time.time() - curr_time
    #         # print("Loop dt = ", dt)

    # kit.stop()
    # kit.shutdown()