#!/usr/bin/python
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class VoiceController():

	def __init__(self, dict):

		#Create the node and set shutdown command
		rospy.init_node('voice_control')
		rospy.on_shutdown(self.shutdown)

		#Initialze output msg
		self.msg = Twist()

		#Initalize publisher
		self.pub = rospy.Publisher("cmd_vel", Twist, queue_size=5)

		#Initialize valid command dict
		self.dict = dict

		rospy.Subscriber("kws_data", String, self.parse_kws)
		rospy.spin()

	def parse_kws(self, kws_data):

		for command in self.dict.keys():
			if kws_data.data.find(command) > -1:
				self.msg = self.dict.get(command)
				break

		self.pub.publish(self.msg)


	def shutdown(self):
		rospy.loginfo("Stopping VoiceController")
		self.pub.publish(Twist())#Publish empty twist message to stop
		rospy.sleep(1)


#Format - command:[x,y,z,r,p,y]
dict = {"forward":Twist((0.5,0,0),(0,0,0)),
		"back":Twist((-0.5,0,0),(0,0,0)), 
		"right":Twist((0,0,0,0),(0,-0.5)),
		"left":Twist((0,0,0),(0,0,0.5)),
		"stop":Twist((0,0,0),(0,0,0))
		}

if __name__ == "__main__":
	VoiceController(dict)