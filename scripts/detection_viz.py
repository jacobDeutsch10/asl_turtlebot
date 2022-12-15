#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker
from asl_turtlebot.msg import DetectedObjectList 
import tf
import numpy as np
from numpy import linalg


class DetVisualizer:

    def __init__(self):
        rospy.init_node('box_viz', anonymous=True)
        rospy.Subscriber('/camera/camera_info', CameraInfo, self.camera_callback)
        rospy.Subscriber("/detector/objects", DetectedObjectList, self.detector_callback)
        self.pub = rospy.Publisher('/boxes', Marker, queue_size=10)
        self.trans_listener = tf.TransformListener()
        self.fov_x = None
        self.fov_y = None
        self.x, self.y, self.theta = 0, 0, 0 
        self.z = .1
        self.detected_objects = {}

        
    def camera_callback(self, data):
        fx, w, fy, h = data.K[0], data.width, data.K[4], data.height
        self.K = data.K#np.reshape(data.K, (3,3))
        self.fx = fx
        self.fy = fy
        self.fov_x = 2*np.arctan2(w, 2*fx)
        self.w = w
        self.h = h
        self.fov_y = 2*np.arctan2(h, 2*fy)
    def detector_callback(self, data:DetectedObjectList):
        for obj, obj_msg in zip(data.objects, data.ob_msgs):
            prev = self.detected_objects.get(obj, {'confidence': 0})
    
            if obj_msg.confidence > prev['confidence'] and obj_msg.confidence > 0.85:
                print('ADDED OBJECT:', obj)
                self.detected_objects[obj] = {
                    'confidence': obj_msg.confidence,
                    'location': self.calculate_location(obj_msg)
                }
    def calculate_location(self, msg):
        x = self.x + np.cos(self.theta)*msg.distance
        y = self.y + np.sin(self.theta)*msg.distance
        z = self.z
        return (x,y,z)
    
    def gen_marker(self):
        ls = Marker()
        ls.id = 4
        ls.header.frame_id = "map"
        ls.type = Marker.SPHERE_LIST
        ls.scale.x = .1
        ls.scale.y = 0.1
        ls.scale.z = 0.1
        ls.color.b = 1.0
        ls.color.a = 1.0
        ls.points = []
        for k, v in self.detected_objects.items():
            p = Point()
            p.x, p.y, p.z = v['location']
            ls.points.append(p)


        
        return ls

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            try:
                (translation, rotation) = self.trans_listener.lookupTransform(
                    "/map", "/base_footprint", rospy.Time(0)
                )
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:
                print(e)
                pass
            marker = self.gen_marker()
            if len(marker.points):
                self.pub.publish(marker)
            rate.sleep()

if __name__ == '__main__':
    ctrl = DetVisualizer()
    ctrl.run()