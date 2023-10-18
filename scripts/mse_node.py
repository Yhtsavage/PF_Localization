#!/usr/bin/python3
#used to calculate the mse between estimated and ground true pose
import rospy
from geometry_msgs.msg import PoseStamped, PoseArray, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from rospy.rostime import Time
import numpy as np


class MSE_Cal(object):
    def __init__(self) -> None:
        self.estimated_posestampe = PoseStamped()
        self.ground_true = Odometry().pose.pose
        self.estimated_pose_sub =rospy.Subscriber('/estimatedpose', PoseStamped, self.Estimated_call_back)
        self.ground_true_Sub = rospy.Subscriber('/base_pose_ground_truth', Odometry, self.Ground_true_call_back)
        #self.Mse_Publisher = rospy.Publisher('/mse', Float64, queue_size=10)
        self.MSE = Float64()
        self.rate = rospy.Rate(5)
    def Ground_true_call_back(self, ground_true):
        self.ground_true = ground_true.pose.pose

    def Estimated_call_back(self, EstPoseStampe):#PoseStamp
        self.estimated_posestampe = EstPoseStampe
        #self.estimated_time = rospy.get_rostime()
        delta_x = abs(self.estimated_posestampe.pose.position.x - self.ground_true.position.x)
        delta_y = abs(self.estimated_posestampe.pose.position.y - self.ground_true.position.y)
        delta_z = abs(self.estimated_posestampe.pose.position.z - self.ground_true.position.z)
        self.MSE.data = (delta_x**2 + delta_y**2 + delta_z**2)**0.5
        rospy.loginfo("MSE: %f, Estimated Time: %s", self.MSE.data, self.estimated_posestampe.header.seq)
        self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('mse_calculator')
    rate = rospy.Rate(5)
    mse_cal = MSE_Cal()
    rospy.spin()
        



