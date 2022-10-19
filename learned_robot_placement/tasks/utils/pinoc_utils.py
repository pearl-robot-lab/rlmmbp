from __future__ import print_function
import numpy as np
import os
import pinocchio as pin
from scipy.spatial.transform import Rotation


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
        move_group: str = "arm_right", # Can be 'arm_right' or 'arm_left'
        include_torso: bool = False, # Use torso in th IK solution
        include_base: bool = False, # Use base in th IK solution
        max_rot_vel: float = 1.0472
    ) -> None:
        # Settings
        self.damp = 1e-10 # Damping co-efficient for linalg solve (to avoid singularities)
        self._include_torso = include_torso
        self._include_base = include_base
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
        if self._include_torso:
            # Add torso joint
            jointsOfInterest = ['torso_lift_joint'] + jointsOfInterest
        if self._include_base:
            # Add base joints
            jointsOfInterest = ['X','Y','R',] + jointsOfInterest # 10 DOF with holo base joints included (11 with torso)

        remove_ids = list()
        for jnt in jointsOfInterest:
            if self.model.existJointName(jnt):
                remove_ids.append(self.model.getJointId(jnt))
            else:
                print('[IK WARNING]: joint ' + str(jnt) + ' does not belong to the model!')
        jointIdsToExclude = np.delete(np.arange(0,self.model.njoints), remove_ids)
        # Lock extra joints except joint 0 (root)
        reference_configuration=pin.neutral(self.model)
        if not self._include_torso:
            reference_configuration[26] = 0.25 # lock torso_lift_joint at 0.25
        self.model = pin.buildReducedModel(self.model, jointIdsToExclude[1:].tolist(), reference_configuration=reference_configuration)
        assert (len(self.model.joints)==(len(jointsOfInterest)+1)), "[IK Error]: Joints != nDoFs"
        self.model_data = self.model.createData()
        # Define Joint-Limits
        self.joint_pos_min = np.array([-1.1780972451, -1.1780972451, -0.785398163397, -0.392699081699, -2.09439510239, -1.41371669412, -2.09439510239])
        self.joint_pos_max = np.array([+1.57079632679, +1.57079632679, +3.92699081699, +2.35619449019, +2.09439510239, +1.41371669412, +2.09439510239])
        if self._include_torso:
            self.joint_pos_min = np.hstack((np.array([0.0]),self.joint_pos_min))
            self.joint_pos_max = np.hstack((np.array([0.35]),self.joint_pos_max))
        if self._include_base:
            self.joint_pos_min = np.hstack((np.array([-100.0, -100.0, -100.0, -100.0]),self.joint_pos_min))
            self.joint_pos_max = np.hstack((np.array([+100.0, +100.0, +100.0, +100.0]),self.joint_pos_max))
        self.joint_pos_mid = (self.joint_pos_max + self.joint_pos_min)/2.0
        # Get End Effector Frame ID
        self.id_EE = self.model.getFrameId(name_end_effector)

    def solve_fk_tiago(self, curr_joints):
        pin.framesForwardKinematics(self.model,self.model_data,curr_joints)
        oMf = self.model_data.oMf[self.id_EE]
        ee_pos = oMf.translation
        ee_quat = pin.Quaternion(oMf.rotation)

        return ee_pos, np.array([ee_quat.w,ee_quat.x,ee_quat.y,ee_quat.z])

    def solve_ik_pos_tiago(self, des_pos, des_quat, curr_joints=None, n_trials=7, dt=0.1, pos_threshold=0.05, angle_threshold=15.*np.pi/180, verbose=False):
        # Get IK positions for tiago robot
        damp = 1e-10
        success = False

        if des_quat is not None:
            # quaternion to rot matrix
            des_rot = Rotation.from_quat(np.array([des_quat[1],des_quat[2],des_quat[3],des_quat[0]])).as_matrix() # Quaternion in scalar last format!!!
            oMdes = pin.SE3(des_rot, des_pos)
        else:
            # 3D position error only
            des_rot = None

        if curr_joints is None:
            q = np.random.uniform(self.joint_pos_min, self.joint_pos_max)
        
        for n in range(n_trials):
            for i in range(800):
                pin.framesForwardKinematics(self.model,self.model_data,q)
                oMf = self.model_data.oMf[self.id_EE]
                if des_rot is None:
                    oMdes = pin.SE3(oMf.rotation, des_pos) # Set rotation equal to current rotation to exclude this error
                dMf = oMdes.actInv(oMf)
                err = pin.log(dMf).vector
                if (np.linalg.norm(err[0:3]) < pos_threshold) and (np.linalg.norm(err[3:6]) < angle_threshold):
                    success = True
                    break
                J = pin.computeFrameJacobian(self.model,self.model_data,q,self.id_EE)
                if des_rot is None:
                    J = J[:3,:] # Only pos errors
                    err = err[:3]
                v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
                q = pin.integrate(self.model,q,v*dt)
                # Clip q to within joint limits
                q = np.clip(q, self.joint_pos_min, self.joint_pos_max)

                if verbose:
                    if not i % 100:
                        print('Trial %d: iter %d: error = %s' % (n+1, i, err.T))
                    i += 1
            if success:
                best_q = np.array(q)
                break
            else:
                # Save current solution
                best_q = np.array(q)
                # Reset q to random configuration
                q = np.random.uniform(self.joint_pos_min, self.joint_pos_max)
        if verbose:
            if success:
                print("[[[[IK: Convergence achieved!]]]")
            else:
                print("[Warning: the IK iterative algorithm has not reached convergence to the desired precision]")
        
        return success, best_q