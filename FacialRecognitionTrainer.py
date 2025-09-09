import numpy as np
import cv2 as cv
from PIL import Image
import os
import pickle
#----------------------------- facial recognition ---------------------------------------------
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_smile.xml')

base_dir = 'C:\Users\kae-w\Desktop\Facial_Recognition\Facial_Recognition\FacialRecognizerAndTrainer' #change this when on robot
recognizer = cv.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(base_dir):
    for files in files:
        if files.endswith("png") or files.endswith("jpg") or files.endswith("jfif"):
            path = os.path.join(root, files)
            label = os.path.basename(root).replace(" ", "_").lower()
            print(os.path.basename(root))
            print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id = label_ids[label]
            print(label_ids)
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            for (x,y,h,w) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainer.yml')

# =============================================================================================

#---------------------------------------- Recognition -----------------------------------------
# Face Cascade here too
# rest in FacialIdentificationTutorial.py and next class
    #Pip install pillow for:
#from PIL  Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")


