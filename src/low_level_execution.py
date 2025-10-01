#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
import time

def send_goal(publisher, position, orientation):
    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.stamp = rospy.Time.now()
    
    goal.pose.position.x = position['x']
    goal.pose.position.y = position['y']
    goal.pose.position.z = position['z']
    
    goal.pose.orientation.x = orientation['x']
    goal.pose.orientation.y = orientation['y']
    goal.pose.orientation.z = orientation['z']
    goal.pose.orientation.w = orientation['w']
    
    rospy.loginfo("Sending goal to position (%.3f, %.3f), orientation z: %.3f, w: %.3f",
                  position['x'], position['y'], orientation['z'], orientation['w'])
    publisher.publish(goal)

if __name__ == "__main__":
    rospy.init_node("send_multiple_goals")
    pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    
    rospy.sleep(1)  # Wait for publisher connection

    goals = [
        {  
            'position': {'x': -0.396, 'y': 0.009, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.335, 'w': 0.942}
        },
    ]

    for goal in goals:
        send_goal(pub, goal['position'], goal['orientation'])
        rospy.sleep(10) 
    rospy.loginfo("All goals sent.")

"""

        {  # Goal 2
            'position': {'x': -0.433, 'y': -1.034, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.935, 'w': 0.354}
        },
        {  # Goal 3
            'position': {'x': 0.094, 'y': -0.896, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.385, 'w': 0.923}
        }
"""