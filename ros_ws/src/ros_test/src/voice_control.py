#!/usr/bin/python
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class VoiceController():

    def __init__(self):

        #Create the node and set shutdown command
        rospy.init_node('voice_control')
        rospy.on_shutdown(self.shutdown)

        #Initialze output msg
        self.msg = Twist()

        #Initalize publisher
        self.pub = rospy.Publisher("cmd_vel", Twist, queue_size=5)

        rospy.Subscriber("kws_data", String, self.parse_kws)
        rospy.spin()

    def parse_kws(self, kws_data):

        if kws_data.data.find('forward') > -1:
            self.msg.linear.x = 0.3
            self.msg.angular.z = 0
        elif kws_data.data.find('back') > -1:
            self.msg.linear.x = -0.3
            self.msg.angular.z = 0
        elif kws_data.data.find('right') > -1:
            self.msg.linear.x = 0.1
            self.msg.angular.z = -1
        elif kws_data.data.find('left') > -1:
            self.msg.linear.x = 0.1
            self.msg.angular.z = 1
        elif kws_data.data.find('stop') > -1:
            self.msg = Twist()

        self.pub.publish(self.msg)


    def shutdown(self):
        rospy.loginfo("Stopping VoiceController")
        self.pub.publish(Twist())#Publish empty twist message to stop
        rospy.sleep(1)

if __name__ == "__main__":
    VoiceController()