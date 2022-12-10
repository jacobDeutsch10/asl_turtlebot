#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker 
import tf
import numpy as np
from numpy import linalg


class FovVisualizer:

    def __init__(self):
        rospy.init_node('fov_viz', anonymous=True)
        rospy.Subscriber('/camera/camera_info', CameraInfo, self.fov_callback)
        self.pub = rospy.Publisher('/FOV',Marker, queue_size=10)
        self.fov_x = None
        self.fov_y = None
        self.x, self.y, self.theta = 0, 0, 0 
        self.z = 0
        self.d = .2
        

    def gazebo_callback(self, data):
        pose = data.pose[data.name.index("turtlebot3_burger")]
        twist = data.twist[data.name.index("turtlebot3_burger")]
        self.x = pose.position.x
        self.y = pose.position.y
        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]
    def fov_callback(self, data):
        fx, w, fy, h = data.K[0], data.width, data.K[4], data.height
        self.K = data.K#np.reshape(data.K, (3,3))
        self.fx = fx
        self.fy = fy
        self.fov_x = 2*np.arctan2(w, 2*fx)
        self.w = w
        self.h = h
        self.fov_y = 2*np.arctan2(h, 2*fy)

    def gen_marker(self):
        ls = Marker()
        ls.id = 4
        ls.header.frame_id = "base_camera"
        ls.type = Marker.LINE_LIST
        ls.scale.x = .02
        ls.color.g = 1.0
        ls.color.a =1.0
        cam = Point()
        cam.z = self.z
        cam.x = self.x
        cam.y = self.y
        r = (self.w-self.K[2])*self.d/self.fx#np.tan(self.fov_x/2)*self.d 
        l = -self.K[2]*self.d/self.fx#-r
        t = (self.h-self.K[5])*self.d/self.fy#np.tan(self.fov_y/2)*self.d
        b = -self.K[5]*self.d/self.fy#-t
        print(r,b,l,t)
        bl = Point()
        bl.x = self.x + self.d
        bl.y = self.y + l
        bl.z = self.z + b

        br = Point()
        br.x = self.d
        br.y = self.y + r
        br.z = self.z + b

        tl = Point()
        tl.x = self.x + self.d
        tl.y = self.y + l
        tl.z = self.z + t
        
        tr = Point()
        tr.x = self.x + self.d
        tr.y = self.y + r
        tr.z = self.z + t
        ls.points = [
            cam,br,
            cam,bl,
            cam,tl,
            cam,tr,
            br, bl,
            br, tr,
            bl, tl,
            tr, tl
        ]
        return ls

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            if self.fov_x:
                marker = self.gen_marker()
                self.pub.publish(marker)
            rate.sleep()

if __name__ == '__main__':
    ctrl = FovVisualizer()
    ctrl.run()