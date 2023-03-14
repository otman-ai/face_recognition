import cv2 
import face_recognition
import os
import pandas as pd
main_dir = "Images/"
images = os.listdir(main_dir)
image_path = []
for img in images:
    image_path.append(main_dir+img)
encodeFaces = {}
for i,image in enumerate(image_path):
    MyImage = face_recognition.load_image_file(image)
    MyImage = cv2.cvtColor(MyImage,cv2.COLOR_BGR2RGB)
    myFace = face_recognition.face_locations(MyImage)[0]
    encodeFaces[images[i].split(".")[0]] = face_recognition.face_encodings(MyImage)[0]
df = pd.DataFrame(encodeFaces)
df.to_csv("faces_encoding.csv")
print("Encoding is completed.")