from __future__ import print_function
import numpy as np
import os
import pinocchio as pin
from scipy.spatial.transform import Rotation
from torch.nn.functional import interpolate


def get_se3_err(pos_first, quat_first, pos_second, quat_second):
    # Retruns 6 dimensional log.SE3 error between two poses expressed as position and quaternion rotation
    
    rot_first = Rotation.from_quat(np.array([quat_first[1],quat_first[2],quat_first[3],quat_first[0]])).as_matrix() # Quaternion in scalar last format!!!
    rot_second = Rotation.from_quat(np.array([quat_second[1],quat_second[2],quat_second[3],quat_second[0]])).as_matrix() # Quaternion in scalar last format!!!
    
    oMfirst = pin.SE3(rot_first, pos_first)
    oMsecond = pin.SE3(rot_second, pos_second)
    firstMsecond = oMfirst.actInv(oMsecond)
    
    return pin.log(firstMsecond).vector # log gives us a spatial vector (exp co-ords)


class PinTiagoIKSolver(object):
    def __init__(
        self,
        urdf_name: str = "tiago_dual_holobase.urdf",
        move_group: str = "arm_right", # Can only be 'arm_right' or 'arm_left'
        max_rot_vel: float = 1.0472
    ) -> None:
        # Settings
        self.damp = 1e-10 # Damping co-efficient for linalg solve (to avoid singularities)
        self.max_rot_vel = max_rot_vel # Maximum rotational velocity of all joints
        
        ## Load urdf
        urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/" + urdf_name
        self.model = pin.buildModelFromUrdf(urdf_file)
        # Choose joints
        name_end_effector = "gripper_"+move_group[4:]+"_grasping_frame"
        # left_7_link to ee tf = [[0., 0.,  1.,  0.      ],
        #                    [ 0., 1.,  0.,  0.      ],
        #                    [-1., 0.,  0., -0.196575],
        #                    [ 0., 0.,  0.,  1.      ]]
        # right_7_link to ee tf = [[0., 0., -1.,  0.      ],
        #                    [0., 1.,  0.,  0.      ],
        #                    [1., 0.,  0.,  0.196575],
        #                    [0., 0.,  0.,  1.      ]]
        # # name_base_link = "base_footprint"#"world"
        jointsOfInterest = [move_group+'_1_joint', move_group+'_2_joint',
                            move_group+'_3_joint', move_group+'_4_joint', move_group+'_5_joint',
                            move_group+'_6_joint', move_group+'_7_joint']
        # Add base joints
        jointsOfInterest = ['X','Y','R',] + jointsOfInterest # 10 DOF with holo base joints included
        
        remove_ids = list()
        for jnt in jointsOfInterest:
            if self.model.existJointName(jnt):
                remove_ids.append(self.model.getJointId(jnt))
            else:
                print('[IK WARNING]: joint ' + str(jnt) + ' does not belong to the model!')
        jointIdsToExclude = np.delete(np.arange(0,self.model.njoints), remove_ids)
        # Lock extra joints except joint 0 (root)
        reference_configuration=pin.neutral(self.model)
        reference_configuration[26] = 0.25 # torso_lift_joint
        self.model = pin.buildReducedModel(self.model, jointIdsToExclude[1:].tolist(), reference_configuration=reference_configuration)
        assert (len(self.model.joints)==(len(jointsOfInterest)+1)), "[IK Error]: Joints != nDoFs"
        self.model_data = self.model.createData()
        # Define Joint-Limits
        self.joint_pos_min = np.array([-100.0, -100.0, -100.0, -100.0, -1.1780972451, -1.1780972451, -0.785398163397, -0.392699081699, -2.09439510239, -1.41371669412, -2.09439510239])
        self.joint_pos_max = np.array([+100.0, +100.0, +100.0, +100.0, +1.57079632679, +1.57079632679, +3.92699081699, +2.35619449019, +2.09439510239, +1.41371669412, +2.09439510239])
        self.joint_pos_mid = (self.joint_pos_max + self.joint_pos_min)/2.0
        # Get End Effector Frame ID
        self.id_EE = self.model.getFrameId(name_end_effector)

    def solve_fk_tiago(self, curr_joints):
        pin.framesForwardKinematics(self.model,self.model_data,curr_joints)
        oMf = self.model_data.oMf[self.id_EE]
        ee_pos = oMf.translation
        ee_quat = pin.Quaternion(oMf.rotation)

        return ee_pos, np.array([ee_quat.w,ee_quat.x,ee_quat.y,ee_quat.z])

    def solve_ik_arm_vel_w_limits_tiago(self, des_pos, des_quat, curr_joints, base_vels, dt, include_base_joints_in_ik=False):
        # Get arm IK velocities with joint limit avoidance and a limit hit check
        
        pin.framesForwardKinematics(self.model,self.model_data,curr_joints)
        oMf = self.model_data.oMf[self.id_EE]
        ee_pos = oMf.translation
        ee_quat = pin.Quaternion(oMf.rotation)
        
        if des_quat is not None:
            # quaternion to rot matrix
            des_rot = Rotation.from_quat(np.array([des_quat[1],des_quat[2],des_quat[3],des_quat[0]])).as_matrix() # Quaternion in scalar last format!!!
        else:
            # 3D pos IK only
            des_rot = oMf.rotation
        oMdes = pin.SE3(des_rot, des_pos)
        dMf = oMdes.actInv(oMf)
        # Optional: interpolate to desired coniguration?
        # dMf = pin.SE3.Interpolate(pin.SE3(1), dMf, 0.1)
        # Get desired EE velocity
        err = pin.log(dMf).vector / dt # log gives us a spatial vector (exp co-ords)
        # Keep the norm of the spatial velocity within a stable value
        norm = np.linalg.norm(err)
        if norm > 2.0:
            err /= norm

        J = pin.computeFrameJacobian(self.model,self.model_data,curr_joints,self.id_EE)
        # Exclude base joints -> Modify Jacobian accordingly
        if not include_base_joints_in_ik:
            J = J[:,3:] # Excluding first three joints
        if des_quat is not None:
            # 6D IK
            ik_velocities = - J.T.dot(np.linalg.solve(J.dot(J.T) + self.damp * np.eye(6), err))
        else:
            J_reduced = J[:3,:] # Only pos errors
            err = err[:3]
            ik_velocities = - J_reduced.T.dot(np.linalg.solve(J_reduced.dot(J_reduced.T) + self.damp * np.eye(3), err))
        
        # Scale/clamp the ik vels based on max_rot_vel of arm joints
        if (abs(ik_velocities) > self.max_rot_vel).any():
            ik_velocities *= (self.max_rot_vel/np.max(ik_velocities))
        # ik_velocities = np.clip(ik_velocities,-self.max_rot_vel,self.max_rot_vel)

        # Compute joint limit avoidance objective
        lamb = 1.0
        joint_limit_vel = lamb*(self.joint_pos_mid[4:] - curr_joints[4:]) # only for arm joints

        # Check for joint limit violations if stepping with dt
        if include_base_joints_in_ik:
            # project into null-space
            ik_velocities[3:] = ik_velocities[3:] + (np.eye(ik_velocities[3:].shape[0]) - np.linalg.pinv(J[:,3:]).dot(J[:,3:])).dot(joint_limit_vel)
            ik_vels=np.array(ik_velocities)
        else:
            ik_velocities = ik_velocities + (np.eye(ik_velocities.shape[0]) - np.linalg.pinv(J).dot(J)).dot(joint_limit_vel)
            ik_vels = np.hstack((base_vels,ik_velocities)) # add velocities for base joints
        q = pin.integrate(self.model,curr_joints,ik_vels*dt)
        limits = ((q < self.joint_pos_min) + (q > self.joint_pos_max))
        q[limits] = curr_joints[limits] # don't move arm joints at limits
        limit_hit = np.sum(limits) > 0
        ik_temp = ik_velocities[-7:]
        ik_temp[limits[-7:]] = 0.0 # zero velocities for joints that are at limits

        # # Debug: No movement if limits hit
        # if limit_hit:
        #     ik_velocities[:] = 0.0
        #     q = curr_joints

        # Update ee_pose and vels
        pin.framesForwardKinematics(self.model,self.model_data,q)
        oMf = self.model_data.oMf[self.id_EE]
        ee_pos = oMf.translation
        ee_quat = pin.Quaternion(oMf.rotation)
        ee_vel = J.dot(ik_velocities)

        return ik_velocities, ee_pos, np.array([ee_quat.w,ee_quat.x,ee_quat.y,ee_quat.z]), ee_vel, limit_hit