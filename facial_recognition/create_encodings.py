#!usr/bin/env python

import argparse
import os
import pickle
import cv2
import face_recognition

#Command Line Implementation
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to image dataset")
ap.add_argument("-e", "--encodings", required=False,
	help="path to save encodings pickle")
args = vars(ap.parse_args())

dataset_path = args["dataset"]
encoding_path = args["encodings"]

knownEncodings = []
knownNames = []

for name in os.listdir(dataset_path):
    data_path = os.path.join(dataset_path, name)

    i = 0
    size = len(os.listdir(data_path))

    for image in os.listdir(data_path):
        image_path = os.path.join(data_path, image)

        print("Processing "+str(i+1)+"/"+str(size))

        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bounding_boxes = face_recognition.face_locations(rgb, model='cnn')
        encodings = face_recognition.face_encodings(rgb, bounding_boxes)

        for encoding in encodings:
            knownNames.append(name)
            knownEncodings.append(encoding)
        i+=1

print("Saving Encodings")
data = {"encodings":knownEncodings, "names":knownNames}
out = open(encoding_path,"wb")
out.write(pickle.dumps(data))
out.close()