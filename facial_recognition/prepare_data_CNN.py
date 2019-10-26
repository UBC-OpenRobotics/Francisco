#!usr/bin/env python

import argparse
import os
import cv2

#Command line implementation
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Dataset path")
ap.add_argument("-s", "--save", required=True, help="save path")
ap.add_argument("-w","--width", required=False, help="width of faces to input (default=32)")
ap.add_argument("-h","--height", required=False, help="height of faces to input (default=32)")
args = vars(ap.parse_args())

dataset_path = args["dataset"]
save_path = args["save"]
one_hot = []

if args['width'] and args["height"]:
	width = args["width"]
	height = args["height"]
else:
	width = 32
	height = 32

train = {"images":[], "labels":[]}

for name in os.listdir(dataset_path):
    one_hot.append(name)
    
for name in one_hot:
    folder_path = os.path.join(dataset_path, name)
    
    print("Processing {}".format(name))
    
    size = len(os.listdir(folder_path))
    c = 0
    for image_name in os.listdir(folder_path):
        c+=1
        print("{}/{}".format)(c,size)
        
        image_dir = os.path.join(folder_path, image_name)
        
        image = cv2.imread(image_dir)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bounding_boxes = face_recognition.face_locations(rgb, model='cnn')

        for top, right, bottom, left in bounding_boxes:
            face = rgb[top:bottom,left:right]
            face = cv2.resize(face,(width,height))
            face = face/255.0
            
            label = np.zeros(len(one_hot))
            i = one_hot.index(name)
            label[i]=1.0
            
            train["images"].append(face)
            train["labels"].append(label)
            

            
data = {"train":train}
out = open(save_path,"wb")
out.write(pickle.dumps(data))
out.close()
print("Wrote {}".format(save_path))