import tensorflow as tf
import os
import numpy as np
import face_recognition
import cv2
import matplotlib.pyplot as plt
import random
from tensorflow.keras import datasets, layers, models
import pickle

def createData(dataset_path, save_path):
    
    train={"images":[],"labels":[]}

    w = 32
    h = 32

    k = 0.1
    for name in os.listdir(dataset_path):
        name_path = os.path.join(dataset_path, name)

        for image_name in os.listdir(name_path):
            image_path = os.path.join(name_path, image_name)
            image = cv2.imread(image_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bounding_boxes = face_recognition.face_locations(rgb, model='cnn')


            for top, right, bottom, left in bounding_boxes:
                face = rgb[top:bottom,left:right]
                face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                face = cv2.resize(face,(w,h))
                face = np.expand_dims(face,axis=2)
                train["images"].append(face)
                if name == "Francisco":
                    train["labels"].append(1)
                else:
                    train["labels"].append(0)


    test = {"images":[], "labels":[]}
    for _ in range(int(k*len(train["labels"]))):
        i,label = random.choice(list(enumerate(train["labels"])))
        image = train["images"][i]
        test["images"].append(image)
        test["labels"].append(label)
        del(train["images"][i])
        del(train["labels"][i])
    
    data = {"train":train, "test":test}
    out = open(save_path,"wb")
    out.write(pickle.dumps(data))
    out.close()

def createModel():

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32,1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    
    return model

def trainModel(train, test, model, save_path):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(np.array(train["images"]),np.array(list(train["labels"])), epochs=10, 
                        validation_data=(np.array(test["images"]), np.array(list(test["labels"]))))
    
    fig,ax = plt.subplots()
    fig.set_facecolor("white")
    ax.plot(history.history['acc'], label='accuracy')
    ax.plot(history.history['val_acc'], label = 'val_accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0.5, 1.1])
    plt.legend(loc='lower right')
    plt.show()
    
    model_path = os.path.join(save_path, 'CNN.h5')
    model.save(model_path) 

def loadData(data_path):

    data = pickle.loads(open(data_path, "rb").read())
    
    train = data["train"]
    test = data["test"]
    
    return train, test

def recognizeFace():

    cnn_path = "/home/francisco/Desktop/Notebooks/OpenRobotics/CNN_model/CNN.h5"
    model = tf.keras.models.load_model(cnn_path)

    knownNames = ["Francisco", "Delila"]

    cam = cv2.VideoCapture(0)

    fx = 0.5
    fy = 0.5

    w = 32
    h = 32

    thresh = 0.5

    while True:
        ret, frame = cam.read()

        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame, (0,0), fx=fx, fy=fy)

            bounding_boxes = face_recognition.face_locations(resized_frame, model='cnn')

            for top, right, bottom, left in bounding_boxes:
                    face = rgb[top:bottom,left:right]
                    face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                    face = cv2.resize(face,(w,h))
                    face = np.expand_dims(face,axis=2)
                    face = np.expand_dims(face, axis=0)

                    prediction = model.predict(face)[0]

                    if max(prediction) < thresh:
                        name = "Unknown"
                    else:
                        i = np.where(prediction == max(prediction))[0][0]
                        name = knownNames[i]

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



dataset_path = "/home/francisco/Desktop/Notebooks/OpenRobotics/dataset/"
save_path = "/home/francisco/Desktop/Notebooks/OpenRobotics/data.pickle"
cnn_path = "/home/francisco/Desktop/Notebooks/OpenRobotics/CNN_model/"


createData(dataset_path, save_path)
train, test = loadData(save_path)
model = createModel()
trainModel(train, test, model, cnn_path)