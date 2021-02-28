## OpenCV Implementation of Face Recognizer
## faceDataset.py
## Python file to create dataset of individual person/face

##Import Libraries
import cv2
import os
import glob
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

## Path to Haarcascades .xml file
face_detector = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
# For each person, enter one numeric face id
face_id = 1
print("\n [**INFO**] Initializing face capture. ...")
# Initialize individual sampling face count
count = 0

## Path to directory with images of person. Please make sure you have only one person
files=glob.glob("/home/pi/Downloads//*.jpg")

for file in files:
    img=cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
cv2.destroyAllWindows()   
print("\n [**INFO**] Done. ...")