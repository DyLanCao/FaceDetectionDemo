import numpy as np
import cv2
import sys

#read input file and convert to grayscale
input_file = sys.argv[1]
img = cv2.imread(input_file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load HaarCascade classifier and detect faces in image
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
)

#Iterate through found faces and write cropped faces out sequentially
i = 0
for (x, y, w, h) in faces:
    crop_img = img[y:y + h, x:x + w]
    i += 1
    filename = 'face' + str(i) + '.jpg'
    cv2.imwrite(filename, crop_img)