#!/usr/bin/env python3

import numpy as np
from cw2q4.youbotKineBase import YoubotKinematicBase


class YoubotKinematicStudent(YoubotKinematicBase):
    def __init__(self):
        super(YoubotKinematicStudent, self).__init__(tf_suffix='student')

        # Set the offset for theta --> This was updated on 22/11/2024. Fill it in with your calculated joint offsets in cw1 if you need testing.
        # the standard joint offsets will be updated soon.
        youbot_joint_offsets = [0, 0, 0, 0, 0] # place holder 0s for now

        # Apply joint offsets to dh parameters
        self.dh_params['theta'] = [theta + offset for theta, offset in
                                   zip(self.dh_params['theta'], youbot_joint_offsets)]

        # Joint reading polarity signs
        self.youbot_joint_readings_polarity = [-1, 1, 1, 1, 1]

    def forward_kinematics(self, joints_readings, up_to_joint=5):
        """This function solve forward kinematics by multiplying frame transformation up until a specified
        frame number. The frame transformation used in the computation are derived from dh parameters and
        joint_readings.
        Args:
            joints_readings (list): the state of the robot joints. In a youbot those are revolute
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematics.
                Defaults to 5.
        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix describing the pose of frame_{up_to_joint}
                w.r.t the base of the robot.
        """
        assert isinstance(self.dh_params, dict)
        assert isinstance(joints_readings, list), "joint readings of type " + str(type(joints_readings))
        assert isinstance(up_to_joint, int)
        assert up_to_joint >= 0
        assert up_to_joint <= len(self.dh_params['a'])

        T = np.identity(4)

	# --> This was updated on 23/11/2023. Feel free to use your own code.

        # Apply offset and polarity to joint readings (found in URDF file)
        joints_readings = [sign * angle for sign, angle in zip(self.youbot_joint_readings_polarity, joints_readings)]

        for i in range(up_to_joint):
            A = self.standard_dh(self.dh_params['a'][i],
                                 self.dh_params['alpha'][i],
                                 self.dh_params['d'][i],
                                 self.dh_params['theta'][i] + joints_readings[i])
            T = T.dot(A)
            
        assert isinstance(T, np.ndarray), "Output wasn't of type ndarray"
        assert T.shape == (4, 4), "Output had wrong dimensions"
        return T

    def get_jacobian(self, joint):
        """Given the joint values of the robot, compute the Jacobian matrix. Coursework 2 Question 4a.
        Reference - Lecture 5 slide 24.

        Args:
            joint (list): the state of the robot joints. In a youbot those are revolute

        Returns:
            Jacobian (numpy.ndarray): NumPy matrix of size 6x5 which is the Jacobian matrix.
        """
        assert isinstance(joint, list)
        assert len(joint) == 5

        # Your code starts here ----------------------------

        T_matrices = []
        T = np.identity(4)

        for i in range(5):
            A = self.standard_dh(self.dh_params['a'][i], self.dh_params['alpha'][i], self.dh_params['d'][i], self.dh_params['theta'][i] + joint[i])
            T = T.dot(A)
            T_matrices.append(T)

        jacobian = np.zeros((6,5))
        z_prev = np.array([0, 0, -1])
        o_prev = np.array([0, 0, 0])

        o_end = T_matrices[-1][:3, 3]

        for i in range(5):
            z_i = T_matrices[i][:3, 2]
            o_i = T_matrices[i][:3, 3]

            jacobian[:3, i] = np.cross(z_prev, o_end - o_prev)

            jacobian[3:, i] = z_prev

            z_prev = z_i
            o_prev = o_i

        # Your code ends here ------------------------------

        assert jacobian.shape == (6, 5)
        return jacobian

    def check_singularity(self, joint):
        """Check for singularity condition given robot joints. Coursework 2 Question 4c.
        Reference Lecture 5 slide 30.

        Args:
            joint (list): the state of the robot joints. In a youbot those are revolute

        Returns:
            singularity (bool): True if in singularity and False if not in singularity.

        """
        assert isinstance(joint, list)
        assert len(joint) == 5
        
        # Your code starts here ----------------------------
        jacobian = self.get_jacobian(joint)
        jacobian_linear = jacobian[:3, :]

        det = np.linalg.det(jacobian_linear @ jacobian_linear.T)
        singularity = np.isclose(det, 0, atol=1e-6)
        # Your code ends here ------------------------------

        assert isinstance(singularity, bool)
        return singularity
