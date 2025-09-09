import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
import os
import pickle
import tkinter as tk
from tkinter import Label
import threading
import time

#--------------------------- Recognition Set Up ------------------------------------------------
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")  # Load trained model
labels = {}
pickle_file_path = 'labels.pickle'
try:
    with open(pickle_file_path, 'rb') as f:
        original_labels = pickle.load(f)
        labels = {v: k for k, v in original_labels.items()}
    print("Label IDs and Names Mapping:")
    for key, value in labels.items():
        print(f"ID: {value}, Name: {key}")
except FileNotFoundError:
    print(f"Error: The file '{pickle_file_path}' does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")

#----------------------------- Facial Recognition Setup ---------------------------------------
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_smile.xml')

#----------------------------- Camera Setup -----------------------------------------------------
cam = cv.VideoCapture(0)

# -------------------------Text Setup ---------------------------------------------------------
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1
text_color = (255, 0, 0)  # Red

#------------------------------Window Setup---------------------------------------------------
window = tk.Tk()
window.title("Face Recognition")
eyes_image = Image.open("eyes.jpg")
eyes_img = ImageTk.PhotoImage(eyes_image)
stranger_image = Image.open("stranger_danger.jpg")
stranger_img = ImageTk.PhotoImage(stranger_image)
label = Label(window, image=eyes_img, text="Initializing...", compound="bottom")
label.pack()

switch_to_danger = False
recognized_name = "Initializing..."
last_detection_time = 0
detection_delay = 2  

# ---------------------------- Detection Function ---------------------------------------------
def detect_faces():
    global switch_to_danger, recognized_name, last_detection_time
    while True:
        ret, img = cam.read()
        if not ret:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # print("faces detected")

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                id_, conf = recognizer.predict(roi_gray)

                if conf >= 69:  # Confidence threshold for recognition might have to tweak this some
                    new_name = labels.get(id_, "Unknown")
                    if recognized_name != new_name:
                        recognized_name = new_name
                        switch_to_danger = False
                        last_detection_time = time.time()
                    
                else:
                    if recognized_name  != "Stranger Danger":
                       recognized_name = "Stranger Danger"
                       switch_to_danger = True
                       last_detection_time = time.time()
                
                

                cv.rectangle(img, (x, y), (x+w, y+h), (177, 73, 238), 2)
                text = recognized_name
                text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
                text_y = y + text_size[1] + 5 + h
                cv.putText(img, text, (x, text_y), font, font_scale, text_color, font_thickness, cv.LINE_AA)

                smiles = smile_cascade.detectMultiScale(roi_gray, 1.5, 5)
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.5, 5)
                
                for (ex, ey, ew, eh) in smiles:
                    cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 255), 2)
                for (ex, ey, ew, eh) in eyes:
                    cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        else:
            if time.time() - last_detection_time > detection_delay:
                recognized_name = "No one detected"
                switch_to_danger = False

        window.after(100, update_gui)
        cv.imshow("Main Camera", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()

# ---------------------------- GUI Update Function --------------------------------------------
def update_gui():
    global switch_to_danger, recognized_name
    if switch_to_danger:
        label.config(image=stranger_img, text="Stranger Danger", compound="bottom")
    else:
        label.config(image=eyes_img, text=recognized_name, compound="bottom")

# Start the face detection in a separate thread to keep the GUI responsive
thread = threading.Thread(target=detect_faces)
thread.daemon = True
thread.start()

# Continuously update GUI
window.after(100, update_gui)

window.mainloop()
