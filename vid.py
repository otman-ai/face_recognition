import cv2 
import face_recognition
import os
import datetime
import time
import pandas as pd

encodeFaces =pd.read_csv("faces_encoding.csv")
print("Encoding is loaded successfuly.")

vid = cv2.VideoCapture("vid.mp4")
if (vid.isOpened()== False): 
  print("Error opening video stream or file")
names = []
while(vid.isOpened()):
    ret, frame_ = vid.read()
    frame = cv2.resize(frame_,(0,0),None,0.25,0.25)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame1 = face_recognition.face_locations(frame)
    strart = time.time()
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
            names.append(encodeMyFace)
    cv2.imshow("image",frame_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
df = pd.DataFrame(data=[names],columns=["names"])
df.to_csv(f"data_registred/{datetime.datetime.now()}.csv")
cv2.destroyAllWindows()
