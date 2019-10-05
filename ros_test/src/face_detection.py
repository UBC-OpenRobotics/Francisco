#! /usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

import cv2
import numpy as np
from cv_bridge import CvBridge
import face_recognition


#define constants
fx = 1
fy = 1

def detect_faces(image):

	#convert ros image to cv image
	bridge = CvBridge()

	frame = bridge.imgmsg_to_cv2(image)

	resized_frame = cv2.resize(frame, (0,0), fx=fx, fy=fy)

	#convert bgr to rgb
	rgb_frame = resized_frame[:,:,::-1]

	#detect faces
	face_locations = face_recognition.face_locations(rgb_frame)

	if face_locations:

		rospy.loginfo('Found face')

		for (top, right, bot, left) in face_locations:
			top = int(top/fy)
			right = int(right/fx) 
			bot = int(bot/fy)
			left = int(left/fx)

			cv2.rectangle(frame, (left, top), (right, bot), (0,0,255), 2)

	cv2.imshow('Face Detection', frame)
	cv2.waitKey(1)



def listener():
	rospy.init_node('face_detection')

	rospy.Subscriber('/robot/camera2/image_raw', Image, detect_faces)

	rospy.spin()

if __name__ == '__main__':
	listener()
