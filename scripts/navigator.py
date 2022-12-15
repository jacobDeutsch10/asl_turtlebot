#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker
import tf
import numpy as np
from numpy import linalg
from utils.utils import wrapToPi
from utils.grids import StochOccupancyGrid2D
from planners import AStar, compute_smoothed_traj
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum
from asl_turtlebot.msg import DetectedObject, DetectedObjectList, Mission
from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3


class Objective(Enum):
    EXPLORE = 0
    RESCUE = 1
    

"""explore_waypoints = [
    (3.4, 2.65, 2),
    (2.7, 2.7, 1.5),
    #(1.79, 2.80, -3.08), # starts down 1st hallway
    (0.6, 2.75, -2.2),
    (0.3, 2.1, -2.8),
    #(0.286, 1.12, -1.39), # sees zebra from afar
    (0.32, 0.35, -2), # sees elephant
    (0.32, 0.35, -0.032),
    (2.29, 0.35, 1.53),
    (2.29, 1.53, 1.53),
    (2.29, 0.36, -0.03),
    (3.09, 1.44, -1.43)
    # (3.22, 2.78, -3.1),
    # (1.79, 2.80, -3.08),
    # (0.72, 2.72, -2.18),
    # (0.286, 2.12, -1.39),
    # (0.286, 1.12, -1.39),
    # (0.32, 0.35, -0.032),
    # (2.29, 0.35, 1.53),
    # (2.29, 1.53, 1.53),
    # (2.29, 0.36, -0.03),
    # (3.09, 1.44, -1.43)
]"""

# old waypoints
explore_waypoints = [
    (3.2224685842650422, 2.779933137939175, -3.0834403952502436),
    (1.787139051775313, 2.8038360096923443, -3.0775447685851915),
    (0.7194789276188762, 2.7161621206115685, -2.1832707992361757),
    (0.28557073807874755, 2.1215929112361462, -1.3927941798813344),
    (0.28557073807874755, 1.1215929112361462, -1.3927941798813344),
    (0.3206471710125774, 0.3529687321869445, -0.03165554729860527),
    (2.2927230743347953, 0.3564668258406674, 1.5304144639499595),
    (2.2927230743347953, 1.53, 1.5304144639499595),
    (2.2927230743347953, 0.3564668258406674, -0.0316),
    #(2.4310619084668295, 2.1494065418675166, 1.442590999393995),
    #(2.4310619084668295, 2.3065418675166, 1.442590999393995),
    #(2.6285839898857115, 2.7521494815826832, -0.019519969431599896),
    #(2.792397235042275, 2.7974956213360014, -0.07422529639414903),
    #(3.4163326288802627, 2.5564788119253223, -1.9549612955214137),
    (3.0938649522625554, 1.4372961376680233, -1.4317021216215275)
]

#   <node pkg="asl_turtlebot" type="navigator.py" name="navigator" output="screen" />
class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """

    def __init__(self):
        rospy.init_node("turtlebot_navigator", anonymous=True)
        self.mode = Mode.IDLE

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None
        self.objective = Objective.EXPLORE
        self.waypoints_visited = 0
        self.num_explore = len(explore_waypoints)
        self.stop_time = 4.2069
        self.stop_start = 0

        self.th_init = 0.0
        self.waypoints = explore_waypoints
        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0, 0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution = 0.1
        self.plan_horizon = 15

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.0, 0.0]

        # Robot limits
        self.v_max = 0.2  # maximum velocity
        self.om_max = 0.4  # maximum angular velocity

        self.v_des = 0.12  # desired cruising velocity
        self.theta_start_thresh = 0.05  # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = (
            0.2  # threshold to be far enough into the plan to recompute it
        )
    
        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.2
        self.at_thresh = 0.02
        self.at_thresh_theta = 0.2

        # trajectory smoothing
        self.spline_alpha = 0.065#5#.15
        self.spline_deg = 3  # cubic spline
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 0.5
        self.kpy = 0.5
        self.kdx = 1.5
        self.kdy = 1.5

        self.planning_fails = 0
        # heading controller parameters
        self.kp_th = 1.5

        self.traj_controller = TrajectoryTracker(
            self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max
        )
        self.pose_controller = PoseController(
            0.0, 0.0, 0.0, self.v_max, self.om_max
        )
        self.heading_controller = HeadingController(self.kp_th, self.om_max)

        self.nav_planned_path_pub = rospy.Publisher(
            "/planned_path", Path, queue_size=10
        )
        self.nav_smoothed_path_pub = rospy.Publisher(
            "/cmd_smoothed_path", Path, queue_size=10
        )
        self.nav_smoothed_path_rej_pub = rospy.Publisher(
            "/cmd_smoothed_path_rejected", Path, queue_size=10
        )
        self.nav_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.goal_pub = rospy.Publisher("/waypoint_goal", Marker, queue_size=10)
        self.bark_pub = rospy.Publisher("/meow_woof", String, queue_size=10)

        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_md_callback)
        rospy.Subscriber("/cmd_nav", Pose2D, self.cmd_nav_callback)
        rospy.Subscriber("/detector/objects", DetectedObjectList, self.detector_callback)
        rospy.Subscriber("/rescue", Mission, self.rescue_callback)
        self.detected_objects = {}
        self.mission = []
        print("finished init")

    def dyn_cfg_callback(self, config, level):
        rospy.loginfo(
            "Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}".format(**config)
        )
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]
        return config

    def detector_callback(self, data:DetectedObjectList):

        for obj, obj_msg in zip(data.objects, data.ob_msgs):
            print("!!!!", obj.lower())
            if obj.lower() == 'dog':
                print("!!!!",'PUBLISHING Woof')
                msg = String()
                msg.data = 'woof'
                self.bark_pub.publish(msg)
            if obj.lower() == 'cat':
                print("!!!!",'PUBLISHING MEOW')
                msg = String()
                msg.data = 'meow'
                self.bark_pub.publish(msg)
            prev = self.detected_objects.get(obj, {'confidence': 0})
            
            if self.objective == Objective.EXPLORE and obj_msg.confidence > prev['confidence']:
                print('ADDED OBJECT:', obj)
                distance = obj_msg.distance/4 if obj_msg.distance > 0.33 else 0
                self.detected_objects[obj] = {
                    'confidence': obj_msg.confidence,
                    'location': (np.cos(self.theta)*distance + self.x, np.sin(self.theta)*distance + self.y, self.theta)
                }
    def rescue_callback(self, data):
        added = False
        for n in data.objects:
            name = n.lower()
            if name in self.mission:
                continue
            if self.detected_objects.get(name):
                print('ADDED Waypoint for :', name, self.detected_objects[name]['location'])
                self.waypoints.append(self.detected_objects[name]['location'])
                self.mission.append(name)
                added = True
        if added:
            self.mission.append('home')
            self.waypoints.append((3.0938649522625554, 1.4372961376680233, -1.4317021216215275))

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        if (
            data.x != self.x_g
            or data.y != self.y_g
            or data.theta != self.theta_g
        ):
            rospy.logdebug(f"New command nav received:\n{data}")
            x_g = data.x
            y_g = data.y
            theta_g = data.theta
            
            if not len(self.waypoints):
                print("NEW WAYPOINT ADDED:",(x_g, y_g, theta_g))
                self.waypoints.append((x_g, y_g, theta_g))
            elif (x_g, y_g, theta_g) != self.waypoints[-1]:
                self.waypoints.append((x_g, y_g, theta_g))
                print("NEW WAYPOINT ADDED:",(x_g, y_g, theta_g))


    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x, msg.origin.position.y)

    def map_callback(self, msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if (
            self.map_width > 0
            and self.map_height > 0
            and len(self.map_probs) > 0
        ):
            self.occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                7,
                self.map_probs,
            )
            if self.x_g is not None:
                # if we have a goal to plan to, replan
                rospy.loginfo("replanning because of new map")
                self.replan()  # new map, need to replan

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)
    def publish_waypoint(self):
        if self.x_g is None:
            return
        marker = Marker()
        marker.header.frame_id = 'odom'
        marker.pose.position.x = self.x_g
        marker.pose.position.y = self.y_g
        marker.pose.position.z = .1
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.b = 1.0
        marker.color.g = 0
        marker.type = Marker.SPHERE
        self.goal_pub.publish(marker)

    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.near_thresh
        )

    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        return (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.at_thresh
            and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta
        )

    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (
            abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh
        )

    def close_to_plan_start(self):
        return (
            abs(self.x - self.plan_start[0]) < self.start_pos_thresh
            and abs(self.y - self.plan_start[1]) < self.start_pos_thresh
        )

    def snap_to_grid(self, x):
        return (
            self.plan_resolution * round(x[0] / self.plan_resolution),
            self.plan_resolution * round(x[1] / self.plan_resolution),
        )

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i, 0]
            pose_st.pose.position.y = traj[i, 1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        t = self.get_current_plan_time()
        if self.mode == Mode.PARK:
            V, om = self.pose_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.TRACK:
            V, om = self.traj_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        else:
            V = 0.0
            om = 0.0

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime() - self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0
    def has_stopped(self):

        return self.mode == Mode.IDLE and \
               rospy.get_rostime() - self.stop_start > rospy.Duration.from_sec(self.stop_time)
    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """
        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo(
                "Navigator: replanning canceled, waiting for occupancy map."
            )
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        problem = AStar(
            state_min,
            state_max,
            x_init,
            x_goal,
            self.occupancy,
            self.plan_resolution,
        )

        rospy.loginfo("Navigator: computing navigation plan")
        success = problem.solve()
        if not success:
            self.planning_fails += 1
            rospy.loginfo("Planning failed")
            if self.planning_fails > 20:
                self.x_g = None
                self.y_g = None
                self.theta_g = None
                self.waypoints.pop(0)
                self.switch_mode(Mode.IDLE)
                if self.objective == Objective.RESCUE and len(self.mission):
                    self.mission.pop(0)
            return
        else:
            # forget about goal:
            self.planning_fails = 0
            rospy.loginfo("Planning Succeeded")
        

        planned_path = problem.path

        # Check whether path is too short
        if len(planned_path) < 4:
            rospy.loginfo("Path too short to track")
            self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
            self.switch_mode(Mode.PARK)
            return

        # Smooth and generate a trajectory
        t_new, traj_new = compute_smoothed_traj(
            planned_path, self.v_des, self.spline_deg, self.spline_alpha, self.traj_dt
        )

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = (
                self.current_plan_duration - self.get_current_plan_time()
            )

            # Estimate duration of new trajectory
            th_init_new = traj_new[0, 2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err / self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                rospy.loginfo(
                    "New plan rejected (longer duration than current plan)"
                )
                self.publish_smoothed_path(
                    traj_new, self.nav_smoothed_path_rej_pub
                )
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0, 2]
        self.heading_controller.load_goal(self.th_init)

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return

        rospy.loginfo("Ready to track")
        self.switch_mode(Mode.TRACK)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # try to get state information to update self.x, self.y, self.theta
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
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print(e)
                pass
            #print(self.mode,(self.x_g, self.y_g, self.theta_g), self.objective)
            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                if len(self.waypoints):
                    if self.objective == Objective.EXPLORE or self.has_stopped():
                        self.x_g, self.y_g, self.theta_g = self.waypoints[0]
                        self.publish_waypoint()
                        print('New Goal: ', (self.x_g, self.y_g, self.theta_g))
                        self.replan()

            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.TRACK:
                if self.near_goal():
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (
                    rospy.get_rostime() - self.current_plan_start_time
                ).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan()  # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.PARK:
                if self.x_g is None:
                    self.switch_mode(Mode.IDLE)
                if self.at_goal():
                    print('reached goal!')
                    self.waypoints_visited +=1
                    if self.waypoints_visited == self.num_explore:
                        self.objective = Objective.RESCUE
                    if self.objective == Objective.RESCUE and len(self.mission):
                        self.mission.pop(0)
                    # forget about goal:
                    self.x_g = None
                    self.y_g = None
                    self.theta_g = None
                    self.waypoints.pop(0)
                    self.stop_start = rospy.get_rostime()
                    self.switch_mode(Mode.IDLE)

            self.publish_control()
            
            rate.sleep()


if __name__ == "__main__":
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    nav.run()
