import cv2 
import face_recognition
import os
from datetime import datetime
import time
import pandas as pd
encodeFaces = pd.read_csv("faces_encoding.csv")

print("Encoding is loaded successfuly.")
vid = cv2.VideoCapture(0)
dict_ = {}
dict_["name"] = []
dict_["id"] = []

while(True):
    ret, frame_ = vid.read()
    frame = cv2.resize(frame_,(0,0),None,0.25,0.25)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame1 = face_recognition.face_locations(frame)
    strart = time.time()
    i = 0
    encodeTestFace = face_recognition.face_encodings(frame,frame1)
    for face_location,face_encode in zip(frame1,encodeTestFace):
        for encodeMyFace in encodeFaces.columns:
            resutls = face_recognition.compare_faces([encodeFaces[encodeMyFace].values],face_encode)[0]
            if resutls:
                y1,x2,y2,x1 = face_location
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(frame_,(x1,y1),(x2,y2),(0, 254, 2),2)
                cv2.rectangle(frame_,(x2,y2),(x1,y2-25),(0, 254, 2),cv2.FILLED)
                cv2.putText(frame_,encodeMyFace,(x1,y2-10),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255))
                dict_["name"].append(encodeMyFace)
                dict_["id"].append(i)
                i+=1
    cv2.imshow("Otman",frame_)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
if not os.path.exists("data_registred/"):
    os.makedirs("data_registred/")
df = pd.DataFrame(dict_)
date = datetime.now()
df.to_csv(f"data_registred/{date.year}-{date.month}-{date.day} {date.hour}-{date.minute}-{date.second}.csv")