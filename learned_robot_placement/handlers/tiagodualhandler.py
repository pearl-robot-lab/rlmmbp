import torch
# import numpy as np
from learned_robot_placement.handlers.base.tiagohandler import TiagoBaseHandler
from learned_robot_placement.robots.articulations.tiago_dual import TiagoDual
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.torch.maths import torch_rand_float

# sensors
# from omni.isaac.isaac_sensor import _isaac_sensor
# from omni.isaac.range_sensor._range_sensor import acquire_lidar_sensor_interface

class TiagoDualHandler(TiagoBaseHandler):
    def __init__(self, sim_config, num_envs, device):
        # self.task = task_class
        self._sim_config = sim_config
        self._num_envs = num_envs
        self._robot_positions = torch.tensor([0, 0, 0]) # placement of the robot in the world
        self._device = device

        # default value, get from config in future
        self.default_zero_env_path = "/World/envs/env_0"

        self.arm_left_start = torch.tensor([0.0, 1.5708, 1.5708,
                                            1.5708, 1.5708, -1.5708, 1.5708], device=self._device)
        self.arm_right_start = torch.tensor([0.0, 1.5708, 1.5708,
                                            1.5708, 1.5708, -1.5708, 1.5708], device=self._device)
        # TODO: get max velo of arm from model
        # currently read from opening model in Isaac sim
        # TODO optimize parameters of model e.g. damping
        self.max_velo_arm = torch.tensor((111.72, 111.72, 134.64, 134.64,
                                          111.72, 100.84, 100.84), device=self._device)

        # articulation View will be created later
        self.robots = None

        #arm dof names
        self._arm_left_names = ["arm_left_1_joint",
                                "arm_left_2_joint",
                                "arm_left_3_joint",
                                "arm_left_4_joint",
                                "arm_left_5_joint",
                                "arm_left_6_joint",
                                "arm_left_7_joint"]

        self._arm_right_names = ["arm_right_1_joint",
                                 "arm_right_2_joint",
                                 "arm_right_3_joint",
                                 "arm_right_4_joint",
                                 "arm_right_5_joint",
                                 "arm_right_6_joint",
                                 "arm_right_7_joint"]

        # TODO: implement getters for gripper dofs or add gripper to arm dofs?
        self._gripper_names = ["gripper_left_left_finger_joint",
                               "gripper_left_right_finger_joint",
                               "gripper_right_left_finger_joint",
                               "gripper_right_right_finger_joint"]


        # contact sensor
        # self._contact_sensor = _isaac_sensor.acquire_contact_sensor_interface()
        # # "/World/envs/env0/Tiago/tiago_dual/wrist_right_ft_tool_link/Contact_Sensor"
        # self._contact_left_path = "/Tiago/wrist_left_ft_tool_link/Contact_Sensor"
        # self._contact_right_path = "/Tiago/wrist_right_ft_tool_link/Contact_Sensor"

        # # imu sensor
        # self._imu_path = "/Tiago/base_imu_link/Imu_Sensor"
        # self.imu_sensor = _isaac_sensor.acquire_imu_sensor_interface()

        # # lidar sensor
        # self._lidar_path = "/Tiago/base_laser_link/Lidar"
        # self.lidar_sensor = acquire_lidar_sensor_interface()

        # values are set in post_reset after model is loaded
        # default_paths = [self._contact_left_path,
        #                  self._contact_right_path,
        #                  self._imu_path,
        #                  self._lidar_path]

        # sensor_paths = self._create_sensor_paths(default_paths)

        # self.contact_left_paths = sensor_paths[0]
        # self.contact_right_paths = sensor_paths[1]
        # self.imu_paths = sensor_paths[2]
        # self.lidar_paths = sensor_paths[3]

        # values are set in post_reset after model is loaded
        self.arm_left_dof_idxs = []
        self.arm_right_dof_idxs = []
        self.gripper_idxs = []

        # dof limits
        self.arm_left_dof_lower = []
        self.arm_left_dof_upper = []
        self.arm_right_dof_lower = []
        self.arm_right_dof_upper = []
        self.gripper_dof_lower = []
        self.gripper_dof_upper = []

    # def _create_sensor_paths(self, default_paths):
    #     start = "/World/envs/env_"
    #     result = []
    #     for path in default_paths:
    #         paths = []
    #         for i in range(self._num_envs):
    #             paths.append(start + str(i) + path)
    #         result.append(paths)
    #     return result

    def get_robot(self):
        # make it in task and use handler as getter for path
        tiago = TiagoDual(prim_path=self.default_zero_env_path + "/TiagoDual", name="TiagoDual",
                          translation=self._robot_positions)
        # Optional: Apply additional articulation settings
        self._sim_config.apply_articulation_settings("TiagoDual", get_prim_at_path(tiago.prim_path),
                                                          self._sim_config.parse_actor_config("TiagoDual"))

    # call it in setup_up_scene in Task
    def create_articulation_view(self):
        self.robots = ArticulationView(prim_paths_expr="/World/envs/.*/TiagoDual", name="tiago_dual_view")
        return self.robots

    def post_reset(self):
        # add dof indexes
        self._set_dof_idxs()
        # set dof limits
        self._set_dof_limits()
        # set new default state for reset
        self._set_default_state()

    def _set_default_state(self):
        joint_states = self.robots.get_joints_default_state()
        jt_pos = joint_states.positions
        jt_pos[:, self.arm_left_dof_idxs] = self.arm_left_start
        jt_pos[:, self.arm_right_dof_idxs] = self.arm_right_start
        self.robots.set_joints_default_state(positions=jt_pos)

    def apply_actions(self, actions):
        # actions = actions.to(self._device)
        # arm_actions=actions[:14]
        # return
        self.apply_arm_action(actions=actions)

    def apply_arm_action(self, actions):
        # Velocity control
        # check action shape

        # split actions
        action_arm_left = actions[:, :7]
        action_arm_right = actions[:, 7:]

        velos = torch.zeros((self.robots.count, self.robots.num_dof), device=self._device)
        velos[:, self.arm_left_dof_idxs] = self.max_velo_arm * action_arm_left
        velos[:, self.arm_right_dof_idxs] = self.max_velo_arm * action_arm_right

        indices = torch.arange(self.robots.count, dtype=torch.int32, device=self._device)
        self.robots.set_joint_velocity_targets(velocities=velos, indices=indices)

    def _set_dof_idxs(self):
        [self.arm_left_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._arm_left_names]
        [self.arm_right_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._arm_right_names]
        [self.gripper_idxs.append(self.robots.get_dof_index(name)) for name in self._gripper_names]
        # [self.base_dof_idx.append(self.robot.get_dof_index(name)) for name in self._base_names]

    def _set_dof_limits(self):
        # (num_envs, num_dofs, 2)
        dof_limits = self.robots.get_dof_limits()
        dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        dof_limits_upper = dof_limits[0, :, 1].to(self._device)

        # set relevant joint limit values
        self.arm_left_dof_lower = dof_limits_lower[self.arm_left_dof_idxs]
        self.arm_left_dof_upper = dof_limits_upper[self.arm_left_dof_idxs]
        self.arm_right_dof_lower = dof_limits_lower[self.arm_right_dof_idxs]
        self.arm_right_dof_upper = dof_limits_upper[self.arm_right_dof_idxs]
        self.gripper_dof_lower = dof_limits_lower[self.gripper_idxs]
        self.gripper_dof_upper = dof_limits_upper[self.gripper_idxs]

    # def get_obs_dict(self):
    #     # create a dict of all sensors and halde the obs buffer in Task with it
    #     # instead of multiple getters
    #     arm_left_pos, arm_right_pos = self.get_arms_dof_pos()
    #     arm_left_vel, arm_right_vel = self.get_arms_dof_vel()
    #     contact_left, contact_right = self.get_contact_sensor_values()
    #     action_arm_left, action_arm_right = self.get_last_action()

    #     obs = {
    #         'arm_left_pos': arm_left_pos,
    #         'arm_right_pos': arm_right_pos,
    #         'arm_left_vel': arm_left_vel,
    #         'arm_right_vel': arm_right_vel,
    #         'action_arm_left': action_arm_left,
    #         'action_arm_right': action_arm_right,
    #         'contact_left_data': contact_left,
    #         "contact_right_data": contact_right,
    #         'imu_data': self.get_imu(),
    #         'lidar_data': self.get_lidar()

    #     }
    #     return obs

    def get_arms_dof_pos(self):
        # (num_envs, num_dof)
        dof_pos = self.robots.get_joint_positions(clone=False)
        # left arm
        arm_left_pos = dof_pos[:, self.arm_left_dof_idxs]
        # right arm
        arm_right_pos = dof_pos[:, self.arm_right_dof_idxs]
        return arm_left_pos, arm_right_pos

    def get_arms_dof_vel(self):
        # (num_envs, num_dof)
        dof_vel = self.robots.get_joint_velocities(clone=False)
        # left arm
        arm_left_vel = dof_vel[:, self.arm_left_dof_idxs]
        # right arm
        arm_right_vel = dof_vel[:, self.arm_right_dof_idxs]
        return arm_left_vel, arm_right_vel

    # from virtual joints
    # def get_base_dof_values(self):
    #     dof_pos = self.robots.get_joint_positions(clone=False)
    #     dof_vel = self.robots.get_joint_velocities(clone=False)
    #     # base later
    #     base_pos = dof_pos[:, self._base_dof_idxs]
    #     base_vel = dof_vel[:, self._base_dof_idxs]
    #     return base_pos, base_vel

    # def get_last_action(self):
    #     # https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html?highlight=prim_paths_expr#omni.isaac.core.utils.types.ArticulationActions
    #     last_action = self.robots.get_applied_actions(clone=True)
    #     action_arm_left = last_action.joint_velocities[:, self.arm_left_dof_idxs]
    #     action_arm_right = last_action.joint_velocities[:, self.arm_right_dof_idxs]
    #     return action_arm_left, action_arm_right

    # # force sensor
    # def get_contact_sensor_values(self):
    #     values_left = np.empty((self._num_envs, 1))
    #     values_right = np.empty((self._num_envs, 1))

    #     for i, (p_left, p_right) in enumerate(zip(self.contact_left_paths, self.contact_right_paths)):
    #         if self._contact_sensor.is_contact_sensor(p_left) and\
    #                 self._contact_sensor.is_contact_sensor(p_left):
    #             read_left = self._contact_sensor.get_sensor_sim_reading(p_left)
    #             read_right = self._contact_sensor.get_sensor_sim_reading(p_right)

    #             values_left[i, :] = read_left.value
    #             values_right[i, :] = read_right.value
    #         else:
    #             print("CONTACT SENSOR NOT FOUND")
    #             break

    #     return values_left, values_right

    # imu sensor
    # def get_imu(self):
    #     # https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.isaac_sensor/docs/index.html
    #     imu_data = np.empty((self._num_envs, 6))

    #     for i, p in enumerate(self.imu_paths):
    #         if self.imu_sensor.is_imu_sensor(p):
    #             read = self.imu_sensor.get_sensor_sim_reading(p)
    #             imu_data[i, 0] = read.lin_acc_x
    #             imu_data[i, 1] = read.lin_acc_y
    #             imu_data[i, 2] = read.lin_acc_z
    #             imu_data[i, 3] = read.ang_vel_x
    #             imu_data[i, 4] = read.ang_vel_y
    #             imu_data[i, 5] = read.ang_vel_z
    #         else:
    #             breakpoint()
    #             print("IMU SENSOR NOT FOUND")
    #             break

    #     return imu_data

    # # lidar sensor
    # # https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.range_sensor/docs/index.html
    # def get_lidar(self):
    #     # 180 degrees = 150 beams
    #     lidar_data = np.empty((self._num_envs, 150))
    #     for i, p in enumerate(self.lidar_paths):
    #         if self.lidar_sensor.is_lidar_sensor(p):
    #             lidar_data[i, :] = self.lidar_sensor.get_linear_depth_data(p).flatten()
    #         else:
    #             print("LIDAR SENSOR NOT FOUND")
    #             break

    #     return lidar_data


    def reset(self, indices, randomize=False):
        joint_states = self.robots.get_joints_default_state()
        jt_pos = joint_states.positions.clone()
        # TODO: add randomization
        # import pdb; pdb.set_trace()
        # noise = torch_rand_float(-1.0, 1.0, jt_pos[:, self.arm_right_dof_idxs].shape, device=self._device)
        # jt_pos[:, self.arm_right_dof_idxs] = noise
        self.robots.set_joint_positions(jt_pos, indices=indices)
