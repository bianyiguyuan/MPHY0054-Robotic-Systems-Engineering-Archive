#!/usr/bin/env python3
import numpy as np
import rospy
import rosbag
import rospkg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cw2q4.youbotKineKDL import YoubotKinematicKDL

import PyKDL
from visualization_msgs.msg import Marker


class YoubotTrajectoryPlanning(object):
    def __init__(self):
        # Initialize node
        rospy.init_node('youbot_traj_cw2', anonymous=True)

        # Save question number for check in main run method
        self.kdl_youbot = YoubotKinematicKDL()

        # Create trajectory publisher and a checkpoint publisher to visualize checkpoints
        self.traj_pub = rospy.Publisher('/EffortJointInterface_trajectory_controller/command', JointTrajectory,
                                        queue_size=5)
        self.checkpoint_pub = rospy.Publisher("checkpoint_positions", Marker, queue_size=100)

    def run(self):
        """This function is the main run function of the class. When called, it runs question 6 by calling the q6()
        function to get the trajectory. Then, the message is filled out and published to the /command topic.
        """
        print("run q6a")
        rospy.loginfo("Waiting 5 seconds for everything to load up.")
        rospy.sleep(2.0)
        traj = self.q6()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = ["arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"]
        self.traj_pub.publish(traj)

    def q6(self):
        """ This is the main q6 function. Here, other methods are called to create the shortest path required for this
        question. Below, a general step-by-step is given as to how to solve the problem.
        Returns:
            traj (JointTrajectory): A list of JointTrajectory points giving the robot joint positions to achieve in a
            given time period.
        """
        # Steps to solving Q6.
        # 1. Load in targets from the bagfile (checkpoint data and target joint positions).
        # 2. Compute the shortest path achievable visiting each checkpoint Cartesian position.
        # 3. Determine intermediate checkpoints to achieve a linear path between each checkpoint and have a full list of
        #    checkpoints the robot must achieve. You can publish them to see if they look correct. Look at slides 39 in lecture 7
        # 4. Convert all the checkpoints into joint values using an inverse kinematics solver.
        # 5. Create a JointTrajectory message.

        # Your code starts here ------------------------------
        target_cart_tf, target_joint_positions = self.load_targets()
        sorted_order, min_dist = self.get_shortest_path(target_cart_tf)
        num_points = 10
        full_checkpoint_tfs = self.intermediate_tfs(sorted_order, target_cart_tf, num_points)
        q_checkpoints = self.full_checkpoints_to_joints(full_checkpoint_tfs, target_joint_positions[:,0])
        traj = JointTrajectory()
        for i in range(q_checkpoints.shape[1]):
            point = JointTrajectoryPoint()
            point.positions = q_checkpoints[:, i].tolist()
            point.time_from_start = rospy.Duration(i*0.5)
            traj.points.append(point)
        # Your code ends here ------------------------------

        assert isinstance(traj, JointTrajectory)
        return traj

    def load_targets(self):
        """This function loads the checkpoint data from the 'data.bag' file. In the bag file, you will find messages
        relating to the target joint positions. You need to use forward kinematics to get the goal end-effector position.
        Returns:
            target_cart_tf (4x4x5 np.ndarray): The target 4x4 homogenous transformations of the checkpoints found in the
            bag file. There are a total of 5 transforms (4 checkpoints + 1 initial starting cartesian position).
            target_joint_positions (5x5 np.ndarray): The target joint values for the 4 checkpoints + 1 initial starting
            position.
        """
        # Defining ros package path
        rospack = rospkg.RosPack()
        path = rospack.get_path('cw2q6')

        # Initialize arrays for checkpoint transformations and joint positions
        target_joint_positions = np.zeros((5, 5))
        # Create a 4x4 transformation matrix, then stack 6 of these matrices together for each checkpoint
        target_cart_tf = np.repeat(np.identity(4), 5, axis=1).reshape((4, 4, 5))

        # Load path for selected question
        bag = rosbag.Bag(path + '/bags/data.bag')
        # Get the current starting position of the robot
        target_joint_positions[:, 0] = self.kdl_youbot.kdl_jnt_array_to_list(self.kdl_youbot.current_joint_position)
        # Initialize the first checkpoint as the current end effector position
        target_cart_tf[:, :, 0] = self.kdl_youbot.forward_kinematics(target_joint_positions[:, 0])

        # Your code starts here ------------------------------
        for topic, msg, t in bag.read_messages(topics=['/target_joint_positions']):
            for i in range(len(msg.points)):
                if i < 4:
                    target_joint_positions[:, i+1] = msg.points[i].positions
        for i in range(1, 5):
            target_cart_tf[:,:,i] = self.kdl_youbot.forward_kinematics(target_joint_positions[:,i])
        print(target_joint_positions)
        print(target_cart_tf)
        # Your code ends here ------------------------------

        # Close the bag
        bag.close()

        assert isinstance(target_cart_tf, np.ndarray)
        assert target_cart_tf.shape == (4, 4, 5)
        assert isinstance(target_joint_positions, np.ndarray)
        assert target_joint_positions.shape == (5, 5)

        return target_cart_tf, target_joint_positions

    def get_shortest_path(self, checkpoints_tf):
        """This function takes the checkpoint transformations and computes the order of checkpoints that results
        in the shortest overall path.
        Args:
            checkpoints_tf (np.ndarray): The target checkpoints transformations as a 4x4x5 numpy ndarray.
        Returns:
            sorted_order (np.array): An array of size 5 indicating the order of checkpoint
            min_dist:  (float): The associated distance to the sorted order giving the total estimate for travel
            distance.
        """

        # Your code starts here ------------------------------
        from itertools import permutations
        def dist(a,b):
            return np.linalg.norm(checkpoints_tf[:3,3,a]-checkpoints_tf[:3,3,b])
        min_dist = float('inf')
        best_order = None
        for perm in permutations([1,2,3,4]):
            order = [0]+list(perm)
            d=0.0
            for i in range(len(order)-1):
                d+=dist(order[i],order[i+1])
            if d<min_dist:
                min_dist=d
                best_order=order
        sorted_order=np.array(best_order)
        # Your code ends here ------------------------------

        assert isinstance(sorted_order, np.ndarray)
        assert sorted_order.shape == (5,)
        assert isinstance(min_dist, float)

        return sorted_order, min_dist

    def publish_traj_tfs(self, tfs):
        """This function gets a np.ndarray of transforms and publishes them in a color coded fashion to show how the
        Cartesian path of the robot end-effector.
        Args:
            tfs (np.ndarray): A array of 4x4xn homogenous transformations specifying the end-effector trajectory.
        """
        id = 0
        for i in range(0, tfs.shape[2]):
            marker = Marker()
            marker.id = id
            id += 1
            marker.header.frame_id = 'base_link'
            marker.header.stamp = rospy.Time.now()
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0 + id * 0.05
            marker.color.b = 1.0 - id * 0.05
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = tfs[0, -1, i]
            marker.pose.position.y = tfs[1, -1, i]
            marker.pose.position.z = tfs[2, -1, i]
            self.checkpoint_pub.publish(marker)

    def intermediate_tfs(self, sorted_checkpoint_idx, target_checkpoint_tfs, num_points):
        """This function takes the target checkpoint transforms and the desired order based on the shortest path sorting, 
        and calls the decoupled_rot_and_trans() function.
        Args:
            sorted_checkpoint_idx (list): List describing order of checkpoints to follow.
            target_checkpoint_tfs (np.ndarray): the state of the robot joints. In a youbot those are revolute
            num_points (int): Number of intermediate points between checkpoints.
        Returns:
            full_checkpoint_tfs: 4x4x(4xnum_points + 5) homogeneous transformations matrices describing the full desired
            poses of the end-effector position.
        """

        # Your code starts here ------------------------------
        full_list=[]
        for i in range(len(sorted_checkpoint_idx)-1):
            A=target_checkpoint_tfs[:,:,sorted_checkpoint_idx[i]]
            B=target_checkpoint_tfs[:,:,sorted_checkpoint_idx[i+1]]
            seg=self.decoupled_rot_and_trans(A,B,num_points)
            if i<(len(sorted_checkpoint_idx)-1):
                full_list.append(seg[:,:,:-1])
            else:
                full_list.append(seg)
        full_checkpoint_tfs=np.concatenate(full_list,axis=2)
        # Your code ends here ------------------------------
       
        return full_checkpoint_tfs

    def decoupled_rot_and_trans(self, checkpoint_a_tf, checkpoint_b_tf, num_points):
        """This function takes two checkpoint transforms and computes the intermediate transformations
        that follow a straight line path by decoupling rotation and translation.
        Args:
            checkpoint_a_tf (np.ndarray): 4x4 transformation describing pose of checkpoint a.
            checkpoint_b_tf (np.ndarray): 4x4 transformation describing pose of checkpoint b.
            num_points (int): Number of intermediate points between checkpoint a and checkpoint b.
        Returns:
            tfs: 4x4x(num_points) homogeneous transformations matrices describing the full desired
            poses of the end-effector position from checkpoint a to checkpoint b following a linear path.
        """

        # Your code starts here ------------------------------
        pos_a=checkpoint_a_tf[:3,3]
        pos_b=checkpoint_b_tf[:3,3]
        R_a=checkpoint_a_tf[:3,:3]
        R_b=checkpoint_b_tf[:3,:3]
        Ra=PyKDL.Rotation(R_a[0,0],R_a[0,1],R_a[0,2],R_a[1,0],R_a[1,1],R_a[1,2],R_a[2,0],R_a[2,1],R_a[2,2])
        Rb=PyKDL.Rotation(R_b[0,0],R_b[0,1],R_b[0,2],R_b[1,0],R_b[1,1],R_b[1,2],R_b[2,0],R_b[2,1],R_b[2,2])
        qa=Ra.GetQuaternion()
        qb=Rb.GetQuaternion()
        total_steps=num_points+2
        tfs=np.zeros((4,4,total_steps))
        for i in range(total_steps):
            alpha=float(i)/(total_steps-1)
            p=(1-alpha)*pos_a+alpha*pos_b
            dot=qa[0]*qb[0]+qa[1]*qb[1]+qa[2]*qb[2]+qa[3]*qb[3]
            if dot<0.0:
                qb=(-qb[0],-qb[1],-qb[2],-qb[3])
                dot=-dot
            if dot>0.9995:
                q_interp=[qa[j]+alpha*(qb[j]-qa[j]) for j in range(4)]
                q_interp=q_interp/np.linalg.norm(q_interp)
            else:
                theta_0=np.arccos(dot)
                sin_theta_0=np.sin(theta_0)
                theta=theta_0*alpha
                sin_theta=np.sin(theta)
                s0=np.cos(theta)-dot*(sin_theta/sin_theta_0)
                s1=sin_theta/sin_theta_0
                q_interp=[s0*qa[j]+s1*qb[j] for j in range(4)]
            q_interp/=np.linalg.norm(q_interp)
            Rint=PyKDL.Rotation.Quaternion(q_interp[0],q_interp[1],q_interp[2],q_interp[3])
            R_mat=np.array([[Rint[0,0],Rint[0,1],Rint[0,2]],[Rint[1,0],Rint[1,1],Rint[1,2]],[Rint[2,0],Rint[2,1],Rint[2,2]]])
            tf=np.eye(4)
            tf[:3,:3]=R_mat
            tf[:3,3]=p
            tfs[:,:,i]=tf
        # Your code ends here ------------------------------

        return tfs

    def full_checkpoints_to_joints(self, full_checkpoint_tfs, init_joint_position):
        """This function takes the full set of checkpoint transformations, including intermediate checkpoints, 
        and computes the associated joint positions by calling the ik_position_only() function.
        Args:
            full_checkpoint_tfs (np.ndarray, 4x4xn): 4x4xn transformations describing all the desired poses of the end-effector
            to follow the desired path.
            init_joint_position (np.ndarray):A 5x1 array for the initial joint position of the robot.
        Returns:
            q_checkpoints (np.ndarray, 5xn): For each pose, the solution of the position IK to get the joint position
            for that pose.
        """
        
        # Your code starts here ------------------------------
        n=full_checkpoint_tfs.shape[2]
        q_checkpoints=np.zeros((5,n))
        q=init_joint_position.copy()
        for i in range(n):
            pose=full_checkpoint_tfs[:,:,i]
            q,err=self.ik_position_only(pose,q)
            q_checkpoints[:,i]=q
        # Your code ends here ------------------------------

        return q_checkpoints

    def ik_position_only(self, pose, q0):
        """This function implements position only inverse kinematics.
        Args:
            pose (np.ndarray, 4x4): 4x4 transformations describing the pose of the end-effector position.
            q0 (np.ndarray, 5x1):A 5x1 array for the initial starting point of the algorithm.
        Returns:
            q (np.ndarray, 5x1): The IK solution for the given pose.
            error (float): The Cartesian error of the solution.
        """
        # Some useful notes:
        # We are only interested in position control - take only the position part of the pose as well as elements of the
        # Jacobian that will affect the position of the error.

        # Your code starts here ------------------------------
        max_iter=100
        epsilon=1e-4
        q=q0.copy()
        desired_pos=pose[:3,3]
        for _ in range(max_iter):
            fk=self.kdl_youbot.forward_kinematics(q)
            current_pos=fk[:3,3]
            error_vec=desired_pos-current_pos
            error=np.linalg.norm(error_vec)
            if error<epsilon:
                break
            J=self.kdl_youbot.get_jacobian(q)
            J_pos=J[0:3,:]
            dq=np.linalg.pinv(J_pos).dot(error_vec)
            q+=dq
        # Your code ends here ------------------------------

        return q, error

    @staticmethod
    def list_to_kdl_jnt_array(joints):
        """This converts a list to a KDL jnt array.
        Args:
            joints (joints): A list of the joint values.
        Returns:
            kdl_array (PyKDL.JntArray): JntArray object describing the joint position of the robot.
        """
        kdl_array = PyKDL.JntArray(5)
        for i in range(0, 5):
            kdl_array[i] = joints[i]
        return kdl_array


if __name__ == '__main__':
    try:
        youbot_planner = YoubotTrajectoryPlanning()
        youbot_planner.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
