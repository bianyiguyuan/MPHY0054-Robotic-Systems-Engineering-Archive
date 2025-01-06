#!/usr/bin/env python

import numpy as np
from cw3q2.iiwa14DynBase import Iiwa14DynamicBase


class Iiwa14DynamicRef(Iiwa14DynamicBase):
    def __init__(self):
        super(Iiwa14DynamicRef, self).__init__(tf_suffix='ref')

    def forward_kinematics(self, joints_readings, up_to_joint=7):
        """This function solve forward kinematics by multiplying frame transformation up until a specified
        joint. Reference Lecture 9 slide 13.
        Args:
            joints_readings (list): the state of the robot joints.
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematics.
                Defaults to 7.
        Returns:
            np.ndarray The output is a numpy 4*4 matrix describing the transformation from the 'iiwa_link_0' frame to
            the selected joint frame.
        """

        assert isinstance(joints_readings, list), "joint readings of type " + str(type(joints_readings))
        assert isinstance(up_to_joint, int)

        T = np.identity(4)
        # iiwa base offset
        T[2, 3] = 0.1575

        # 1. Recall the order from lectures. T_rot_z * T_trans * T_rot_x * T_rot_y. You are given the location of each
        # joint with translation_vec, X_alpha, Y_alpha, Z_alpha. Also available are function T_rotationX, T_rotation_Y,
        # T_rotation_Z, T_translation for rotation and translation matrices.
        # 2. Use a for loop to compute the final transformation.
        for i in range(0, up_to_joint):
            T = T.dot(self.T_rotationZ(joints_readings[i]))
            T = T.dot(self.T_translation(self.translation_vec[i, :]))
            T = T.dot(self.T_rotationX(self.X_alpha[i]))
            T = T.dot(self.T_rotationY(self.Y_alpha[i]))

        assert isinstance(T, np.ndarray), "Output wasn't of type ndarray"
        assert T.shape == (4, 4), "Output had wrong dimensions"

        return T

    def get_jacobian_centre_of_mass(self, joint_readings, up_to_joint=7):
        """Given the joint values of the robot, compute the Jacobian matrix at the centre of mass of the link.
        Reference - Lecture 9 slide 14.

        Args:
            joint_readings (list): the state of the robot joints.
            up_to_joint (int, optional): Specify up to what frame you want to compute the Jacobian.
            Defaults to 7.

        Returns:
            jacobian (numpy.ndarray): The output is a numpy 6*7 matrix describing the Jacobian matrix defining at the
            centre of mass of a link.
        """
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7

        # Your code starts here ----------------------------
        jacobian = np.zeros((6, 7))

        T_com = self.forward_kinematics_centre_of_mass(joint_readings, up_to_joint)
        p_com = T_com[0:3, 3]

        for i in range(up_to_joint):
            T_i = self.forward_kinematics(joint_readings, i)
            z_i = T_i[0:3, 2]        
            p_i = T_i[0:3, 3]         

            Jv_i = np.cross(z_i, (p_com - p_i))
            Jw_i = z_i

            jacobian[0:3, i] = Jv_i
            jacobian[3:6, i] = Jw_i
        # Your code ends here ------------------------------

        assert jacobian.shape == (6, 7)
        return jacobian

    def forward_kinematics_centre_of_mass(self, joints_readings, up_to_joint=7):
        """This function computes the forward kinematics up to the centre of mass for the given joint frame.
        Reference - Lecture 9 slide 14.
        Args:
            joints_readings (list): the state of the robot joints.
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematicks.
                Defaults to 5.
        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix describing the pose of frame_{up_to_joint} for the
            centre of mass w.r.t the base of the robot.
        """
        T= np.identity(4)
        T[2, 3] = 0.1575

        T = self.forward_kinematics(joints_readings, up_to_joint-1)
        T = T.dot(self.T_rotationZ(joints_readings[up_to_joint-1]))
        T = T.dot(self.T_translation(self.link_cm[up_to_joint-1, :]))

        return T

    def get_B(self, joint_readings):
        """Given the joint positions of the robot, compute inertia matrix B.
        Args:
            joint_readings (list): The positions of the robot joints.

        Returns:
            B (numpy.ndarray): The output is a numpy 7*7 matrix describing the inertia matrix B.
        """
        B = np.zeros((7, 7))
        
	    # Your code starts here ------------------------------
        for i in range(1, len(joint_readings) + 1):
            jacobian = self.get_jacobian_centre_of_mass(joint_readings, i)

            J_p = jacobian[0:3, :]
            J_o = jacobian[3:, :]
            m_i = self.mass[i - 1]  
            I_link = np.diag(self.Ixyz[i - 1])  
            
            T_com_i = self.forward_kinematics_centre_of_mass(joint_readings, up_to_joint=i)
            R_i = T_com_i[0:3, 0:3]
            I_base = R_i @ I_link @ R_i.T

            linear_contribution = m_i * (J_p.T @ J_p)
            angular_contribution = J_o.T @ I_base @ J_o
            B += linear_contribution + angular_contribution
        # Your code ends here ------------------------------
        
	    return B

    def get_C_times_qdot(self, joint_readings, joint_velocities):
        """Given the joint positions and velocities of the robot, compute Coriolis terms C.
        Args:
            joint_readings (list): The positions of the robot joints.
            joint_velocities (list): The velocities of the robot joints.

        Returns:
            C (numpy.ndarray): The output is a numpy 7*1 matrix describing the Coriolis terms C times joint velocities.
        """
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7
        assert isinstance(joint_velocities, list)
        assert len(joint_velocities) == 7

        # Your code starts here ------------------------------
        delta = 1e-8
        n = len(joint_readings)

        # Use Christoffel 
        h_ijk = np.zeros((n, n, n))
        C = np.zeros((n, n))   

        for k in range(n):
            delta_q_k = np.zeros(n)
            delta_q_k[k] = delta

            for j in range(n):
                for i in range(n):
                    delta_q_i = np.zeros(n)
                    delta_q_i[i] = delta

                    dB_ij_dqk = (
                        self.get_B((np.array(joint_readings) + delta_q_k).tolist())[i, j] -
                        self.get_B(joint_readings)[i, j]
                    ) / delta

                    dB_jk_dqi = (
                        self.get_B((np.array(joint_readings) + delta_q_i).tolist())[j, k] -
                        self.get_B(joint_readings)[j, k]
                    ) / delta

                    h_ijk[i, j, k] = dB_ij_dqk - 0.5 * dB_jk_dqi

        # Coriolis matrix C
        for k in range(n):
            C += h_ijk[:, :, k] * joint_velocities[k]

        C = C @ joint_velocities
        # Your code ends here ------------------------------

        assert isinstance(C, np.ndarray)
        assert C.shape == (7,)
        return C

    def get_G(self, joint_readings):
        """Given the joint positions of the robot, compute the gravity matrix g.
        Args:
            joint_readings (list): The positions of the robot joints.

        Returns:
            G (numpy.ndarray): The output is a numpy 7*1 numpy array describing the gravity matrix g.
        """
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7

        # Your code starts here ------------------------------
        g = np.zeros(7)
        grav = np.array([0, 0, -self.g])
        jacobian_cache = {}

        for i in range(7):
            if i + 1 not in jacobian_cache:
                jacobian_cache[i + 1] = self.get_jacobian_centre_of_mass(joint_readings, i + 1)

            J_com_i = jacobian_cache[i + 1]
            m_i = self.mass[i]
            F_i = np.concatenate((m_i * grav, np.zeros(3)))  
            tau_i = J_com_i.T @ F_i 
            g += tau_i
        # Your code ends here ------------------------------

        assert isinstance(g, np.ndarray)
        assert g.shape == (7,)
        return g
