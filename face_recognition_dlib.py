import face_recognition
import pickle
import os
import cv2
import numpy as np

def createDataPerson(name, dataset_path):
    folder_path = os.path.join(dataset_path, name)
    os.mkdir(folder_path)
    
    cam = cv2.VideoCapture(0)
    
    frame_skip = 10
    frame_counter = 0
    num_photos = 50
    photo_counter = 0
    
    while(photo_counter < num_photos):
        
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
    
def createEncodings(dataset_path, encoding_path):
    knownEncodings = []
    knownNames = []

    #dataset_path = "/home/francisco/Desktop/Notebooks/OpenRobotics/dataset/"
    #encoding_path = "/home/francisco/Desktop/Notebooks/OpenRobotics/encodings.pickle"

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

def recognizeFace(encoding_path):
    #encoding_path = "/home/francisco/Desktop/Notebooks/OpenRobotics/encodings.pickle"

    data = pickle.loads(open(encoding_path, "rb").read())

    cam = cv2.VideoCapture(0)

    fx = 0.25
    fy = 0.25

    while True:
        ret, frame = cam.read()

        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame, (0,0), fx=fx, fy=fy)

            bounding_boxes = face_recognition.face_locations(resized_frame, model='cnn')
            encodings = face_recognition.face_encodings(resized_frame, bounding_boxes)
            names =[]


        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name="unknown"

            if True in matches:
                indexes = np.where(matches)
                counts={}

                for i in indexes[0]:
                    name = data["names"][i]
                    counts[name] = counts.get(name,0)+1

                name = max(counts, key=counts.get)
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
    cam.release()
    cv2.destroyAllWindows()