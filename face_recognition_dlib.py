#!/usr/bin/env python

import face_recognition
import pickle
import os
import cv2
import numpy as np
import argparse

#Command line implementation
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to load/save encodings")
ap.add_argument("-m", "--maxpeople", required=False,
	help="maximum number of people to recognize (default=5)")
args = vars(ap.parse_args())

if args["maxpeople"]:
	MAXPEOPLE = args["maxpeople"]
else:
	MAXPEOPLE = 5

#Directory to store encodings and name pickle
encoding_path = args["encodings"]

#If encodings already exist, load them in
if os.path.exists(encoding_path):
    data = pickle.loads(open(encoding_path, "rb").read())
    knownNames = data["names"]
    knownEncodings = data["encodings"]
    numPeople = len(knownNames)
    print("Loaded {}".format(encoding_path))
else:
    knownNames = []
    knownEncodings = []
    numPeople = 0

cam = cv2.VideoCapture(0)

fx = 0.5
fy = 0.5

while True:
    ret, frame = cam.read()

    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame, (0,0), fx=fx, fy=fy)

        #Get all faces in resized frame and their encodings
        bounding_boxes = face_recognition.face_locations(resized_frame, model='cnn')
        encodings = face_recognition.face_encodings(resized_frame, bounding_boxes)
        names =[]

        #Get any matches between knownEncodings and extracted encodings
        for encoding in encodings:
            matches = face_recognition.compare_faces(knownEncodings, encoding)
            name="unknown"

            if True in matches:
                indexes = np.where(matches)
                counts={}

                for i in indexes[0]:
                    name = knownNames[i]
                    counts[name] = counts.get(name,0)+1
                name = max(counts, key=counts.get)
            else:
                #If the max number of people has not been reached and the face is unknown, save the encoding and name
                if numPeople < MAXPEOPLE:
                    name = "Person{}".format(numPeople)
                    knownNames.append(name)
                    knownEncodings.append(encoding)
                    numPeople+=1
                    
            names.append(name)


    for (top, right, bottom, left), name in zip(bounding_boxes, names):
        top = int(top/fx)
        right = int(right/fy)
        bottom = int(bottom/fx)
        left = int(left/fy)

        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)

        #Add label
        cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0,0,255), cv2.FILLED)
        #Add name
        cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255),1)

    cv2.imshow('Video', frame)
    #Wait for esc to quit video feed
    if cv2.waitKey(1) == 27:
        break

#Write encodings to encodingpath
data = {"encodings":knownEncodings, "names":knownNames}
out = open(encoding_path,"wb")
out.write(pickle.dumps(data))
out.close()
print("Wrote {}".format(encoding_path))

cam.release()
cv2.destroyAllWindows()