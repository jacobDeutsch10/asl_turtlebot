#!/usr/bin/env python3
import argparse
import rospy
from asl_turtlebot.msg import Mission
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('objects',type=str, nargs='+', help='objects to be rescued')

if __name__ == "__main__":
    args = parser.parse_args()
    mission = Mission()
    mission.objects = args.objects
    print(mission)
    out = ", ".join(args.objects)
    print(['rostopic', 'pub',  '/rescue',  'asl_turtlebot/Mission', '--', f'[{out}]'])
    subprocess.run(['rostopic', 'pub', '/rescue', 'asl_turtlebot/Mission', '--', f'[{out}]'])
