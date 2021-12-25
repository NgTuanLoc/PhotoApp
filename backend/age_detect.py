import cv2 as cv
import numpy as np
import os
from tensorflow import keras


#  ====================== AGE DETECTION ======================
def convertToRGB(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
haar_cascade_face = cv.CascadeClassifier('./model/haarcascade_frontalface_alt2.xml')


#  Let us create a generalised function for the entire face detection process.


def detect_faces(cascade, test_image, scaleFactor = 1.1, face_id=0):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()
    
    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
    
    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors = 5)
    for i, (x, y, w, h) in enumerate(faces_rect):
        cv.imwrite("./data/faces/face_{i}.png".format(i=face_id), image_copy[y:y+h, x:x+w])

    # (x, y, w, h) = faces_rect
    # cv.imwrite("./data/faces/face_{i}.png".format(i=face_id), image_copy[y:y+h, x:x+w])
        # cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)
        
    return image_copy

def read_image(image_path, face_id):
    faces_image = []
    image=cv.imread(image_path)
    detect_faces(haar_cascade_face, image, face_id=face_id)

    try:
        temp_image = cv.imread("./data/faces/face_{0}.png".format(face_id)) 
        temp_image =  cv.cvtColor(temp_image, cv.COLOR_BGR2RGB)
        temp_image =  cv.resize(temp_image,(48,48))
        faces_image.append(temp_image)
    except:
        print("Can not detect face !!!")
        return []
    return faces_image

def predict_image(image):
    model = keras.models.load_model('./model/Age_sex_detection.h5')
    image_f = np.array([image])/255    
    pred_1=model.predict(image_f)
    sex_f=['Male','Female']
    age=int(np.round(pred_1[1][0]))
    sex=int(np.round(pred_1[0][0]))
    print("Predicted Age: "+ str(age))
    print("Predicted Sex: "+ sex_f[sex])
    return [str(age), str(sex_f[sex])]

def age_detection(image_path, face_id):
    faces_info = []
    faces = read_image(image_path, face_id=face_id)
    print("==========================MODEL==========================")

    for i in faces:
        faces_info.append(predict_image(i))

    return faces_info