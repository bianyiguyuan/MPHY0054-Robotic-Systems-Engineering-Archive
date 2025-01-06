#!/usr/bin/env python3
import rospy
import rosbag
import rospkg
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from numpy.linalg import inv

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from cw3q2.iiwa14DynKDL import Iiwa14DynamicKDL 


class IiwaPlanner(object):
    def __init__(self):
        # Initialize ROS
        rospy.init_node("iiwa_planner_node", anonymous=True)
        
        self.trajectory_pub = rospy.Publisher(
            "/iiwa/EffortJointInterface_trajectory_controller/command",
            JointTrajectory,
            queue_size=10
        )

        self.joint_states_sub = rospy.Subscriber(
            "/iiwa/joint_states",
            JointState,
            self._on_joint_state
        )

        self.kdl_model = Iiwa14DynamicKDL()
        self.pkg_path = rospkg.RosPack().get_path('cw3q5')

        self.csv_dir = os.path.join(self.pkg_path, "src")
        self._accel_file = open(os.path.join(self.csv_dir, 'accel_log.csv'), 'w', newline='')
        self._time_file = open(os.path.join(self.csv_dir, 'time_log.csv'), 'w', newline='')
        self.accel_writer = csv.writer(self._accel_file)
        self.time_writer = csv.writer(self._time_file)

        self.plot_threshold = 37.0

        rospy.loginfo("IiwaPlanner initialized.")

    def run(self):
        rospy.loginfo("Waiting for environment setup...")
        rospy.sleep(2.0)
        
        trajectory = self._create_trajectory_from_bag()
        self.trajectory_pub.publish(trajectory)
        rospy.loginfo("Trajectory published.")

        rospy.spin()

    def _create_trajectory_from_bag(self):
        traj_msg = JointTrajectory()
        traj_msg.joint_names = [
            "iiwa_joint_1", "iiwa_joint_2", "iiwa_joint_3",
            "iiwa_joint_4", "iiwa_joint_5", "iiwa_joint_6", "iiwa_joint_7"
        ]

        bag_file_path = os.path.join(self.pkg_path, "bag", "cw3q5.bag")
        try:
            with rosbag.Bag(bag_file_path, 'r') as bag:
                for topic, msg, t in bag.read_messages():
                    for point in msg.points:
                        new_point = JointTrajectoryPoint()
                        new_point.positions = point.positions
                        new_point.velocities = point.velocities
                        new_point.accelerations = point.accelerations
                        new_point.time_from_start = rospy.Duration(10 * (len(traj_msg.points) + 1))
                        traj_msg.points.append(new_point)

            rospy.loginfo("Bag file read successfully. Got %d trajectory points.", len(traj_msg.points))
        except Exception as e:
            rospy.logerr("Error reading bag file: %s", e)
            return traj_msg

        traj_msg.header.stamp = rospy.Time.now()
        return traj_msg

    def _on_joint_state(self, joint_state_msg):
        curr_time = rospy.get_time()  
        positions = joint_state_msg.position
        velocities = joint_state_msg.velocity
        efforts = joint_state_msg.effort
        accelerations = self._compute_acceleration(positions, velocities, efforts)

        self.time_writer.writerow([curr_time])
        self.accel_writer.writerow(accelerations.flatten().tolist())

        if curr_time > self.plot_threshold:
            rospy.loginfo("Time %.2f > threshold %.2f, starting plot...", curr_time, self.plot_threshold)
            self._accel_file.close()
            self._time_file.close()
            self._plot_accelerations()
            rospy.signal_shutdown("Plotting complete. Shutting down node.")

    def _compute_acceleration(self, q, qdot, tau):
        # Forward kinetics
        B_matrix = np.array(self.kdl_model.get_B(q))
        Cq_dot = np.array(self.kdl_model.get_C_times_qdot(q, qdot))
        G_vec = np.array(self.kdl_model.get_G(q))

        # ddq = inv(B) @ (tau - (Cq_dot + G_vec))
        ddq = inv(B_matrix).dot(np.array(tau) - Cq_dot - G_vec)
        return ddq.reshape(-1, 1)  

    def _plot_accelerations(self):
        time_log_path = os.path.join(self.csv_dir, 'time_log.csv')
        accel_log_path = os.path.join(self.csv_dir, 'accel_log.csv')

        times = []
        accel_data = [[] for _ in range(7)] 

        with open(time_log_path, 'r') as tfile:
            treader = csv.reader(tfile)
            for row in treader:
                if row:
                    times.append(float(row[0]))

        with open(accel_log_path, 'r') as afile:
            areader = csv.reader(afile)
            for row in areader:
                if len(row) == 7:
                    for j in range(7):
                        accel_data[j].append(float(row[j]))

        plt.figure()
        for joint_idx in range(7):
            plt.plot(times, accel_data[joint_idx], label=f"Joint_{joint_idx+1}")
        plt.title("IIWA Joint Accelerations Over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration [rad/s^2]")
        plt.legend()
        plt.grid(True)
        plt.show()
        rospy.loginfo("Plotting finished.")


if __name__ == "__main__":
    try:
        planner = IiwaPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
