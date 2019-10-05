#!usr/bin/env python

import argparse
import cv2
import os

#Command line implementation
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
	help="Label to give data")
ap.add_argument("-d", "--dataset", required=True,
	help="dataset directory")
ap.add_argument("-np", "--numphotos", required=False,
	help="number of photos to take (default = 50)")
args = vars(ap.parse_args())

if args["numphotos"]:
	NUMPHOTOS = 50
else:
	NUMPHOTOS = 50

dataset_path["dataset"]
name = args["name"]

folder_path = os.path.join(dataset_path, name)
    
cam = cv2.VideoCapture(0)

frame_skip = 10
frame_counter = 0
photo_counter = 0

#If person already exists, add data to folder
if os.path.exists(folder_path):
    photo_counter=len(os.listdir(folder_path))
    num_photos+=len(os.listdir(folder_path))
else:
    os.mkdir(folder_path)

while(photo_counter < NUMPHOTOS):
    
    ret, frame = cam.read()
    
    if ret and frame_counter==frame_skip:
        frame_counter = 0
        photo_counter+=1
        photo_path = os.path.join(folder_path, name+str(photo_counter)+".jpg")
        cv2.imwrite(photo_path, frame)
        print("Wrote " + photo_path)
    else:
        frame_counter+=1
    
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

print("Complete")

cam.release()
cv2.destroyAllWindows()