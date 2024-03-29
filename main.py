import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import math
import os
import pickle
import time
model= YOLO("C:\mac\pycham\pythonProject3\yolov8m-face.pt")
start_time_1 = time.time()
start_time_2= time.time()
"""--------------  for data set  ---------------"""
frame= cv2.VideoCapture("C:\mac\pycham\pythonProject3\cr.mp4")
my_file =open("C:\mac\pycham\pythonProject3\coco.txt","r")
df= my_file.read()
classNames= df.split("\n")


def load_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(data)
            area3, area4 = data

            return area3,area4
    else:
        print("File doesn't exist.")
        return None

file_path_2="C:\mac\pycham\pythonProject3\counter.pickle"
file_path="C:\mac\pycham\pythonProject3\queue.pickle"
file_path_3="C:\mac\pycham\pythonProject3\\finished.pickle"
area1, area2= load_data(file_path)
area3, area4= load_data(file_path_2)
area5, area6= load_data(file_path_3)

count=0

while True:
    ret,cam =  frame.read()
    if not ret:
        frame.set(cv2.CAP_PROP_POS_FRAMES, 0)

        continue
    count += 1
    if count % 3 != 0:
        continue
    #cam = cv2.resize(cam, (1020, 500))

    result=model.predict(cam)
    list1=[]
    faces_detected_1 = False
    faces_detected_2 = False

    a = result[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list1 = []
    list2= []
    cvzone.putTextRect(cam, 'Queue Timing: ', (454, 475), font=cv2.FONT_HERSHEY_SIMPLEX,
                    colorT=(255, 255, 255),colorR=(0, 0, 0), scale=1, thickness=1)
    cvzone.putTextRect(cam, 'Queue Timing: ', (970, 475), font=cv2.FONT_HERSHEY_SIMPLEX,
                       colorT=(255, 255, 255), colorR=(0, 0, 0), scale=1, thickness=1)

    cvzone.putTextRect(cam, f'Please Move to: counter: ', (50, 260), 2, 2)


    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = classNames[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        w, h = x2 - x1, y2 - y1

        result1 =  cv2.pointPolygonTest(np.array(area1, np.int32),((cx,cy)),False)
        result2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
        result3 = cv2.pointPolygonTest(np.array(area3, np.int32), ((cx, cy)), False)
        result4 = cv2.pointPolygonTest(np.array(area4, np.int32), ((cx, cy)), False)
        result5 = cv2.pointPolygonTest(np.array(area5, np.int32), ((cx, cy)), False)
        result6 = cv2.pointPolygonTest(np.array(area6, np.int32), ((cx, cy)), False)

        print("result3: ",result3)


        if result1 >=0 and result3 <0:
            cvzone.putTextRect(cam, f'waiting', (x1, y1), 1, colorR=(0, 0, 255), thickness=1, colorT=(255, 255, 255))

        if result2 >= 0 and result4 < 0:
            cvzone.putTextRect(cam, f'waiting', (x1, y1), 1, colorR=(0, 0, 255), thickness=1, colorT=(255, 255, 255))


        if result1 >=0:

            cvzone.cornerRect(cam, (x1, y1, w, h), 3, 2)
            cv2.circle(cam, (cx, cy), 4, (255, 0, 0), -1)
            #cvzone.putTextRect(cam, f'waiting', (x1, y1), 1, 1)
            list1.append(cx)

        if result2 >=0:

            cvzone.cornerRect(cam, (x1, y1, w, h), 3, 2)
            cv2.circle(cam, (cx, cy), 4, (255, 0, 0), -1)
            #cvzone.putTextRect(cam, f'waiting', (x1, y1), 1,colorR=(0, 0, 0), thickness=1,colorT=(255, 255, 255))
            list2.append(cx)


        if result3 >=0:
            faces_detected_1 = True
            elapsed_time_1 = time.time() - start_time_1
            elapsed_time_1 = round(elapsed_time_1, 2)

            cvzone.putTextRect(cam, f'Processing', (x1, y1), 1,colorR=(0, 255, 0), thickness=1,colorT=(0, 0, 0))

            # cvzone.putTextRect(cam, f'{elapsed_time_1} Sec', (max(0, x1), max(35, y2)),
            #                    scale=1, thickness=1)
            if elapsed_time_1 != 0:
                cvzone.putTextRect(cam, f'{elapsed_time_1} Sec', (454 + 230, 475), font=cv2.FONT_HERSHEY_SIMPLEX,
                                   colorT=(255, 255, 255), colorR=(0, 0, 0), scale=1, thickness=1)
        if result4 >=0:
            faces_detected_2=True
            elapsed_time_2 = time.time() - start_time_2
            elapsed_time_2 = round(elapsed_time_2, 2)
            cvzone.putTextRect(cam, f'Processing', (x1, y1), 1, colorR=(0, 255, 0),thickness=1,colorT=(0, 0, 0))
            if elapsed_time_2 != 0:
                cvzone.putTextRect(cam, f'{elapsed_time_2} Sec', (970 + 230, 475), font=cv2.FONT_HERSHEY_SIMPLEX,
                                   colorT=(255, 255, 255), colorR=(0, 0, 0), scale=1, thickness=1)

        if result5 >= 0:
            cvzone.putTextRect(cam, f'Purchased', (x1, y1), 1,colorR=(0, 255, 255), thickness=1,colorT=(0, 0, 0))

        if result6 >= 0:
            cvzone.putTextRect(cam, f'Purchased', (x1, y1), 1, colorR=(0, 255, 255), thickness=1, colorT=(0, 0, 0))

    if not faces_detected_1:
        start_time_1 = time.time()
        elapsed_time_1 = 0

    if not faces_detected_2:
        start_time_2 = time.time()
        elapsed_time_2=0

    counter1 = len(list1)
    counter2 = len(list2)
    cvzone.putTextRect(cam, f'counter 1: {counter1}', (50, 60),2,2)
    cvzone.putTextRect(cam, f'counter 2: {counter2}', (50, 160), 2, 2)
    cv2.polylines(cam, [np.array(area1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(cam, [np.array(area2, np.int32)], True, (0, 0, 255), 2)

    if counter1 > counter2:
        cvzone.putTextRect(cam, f'2', (50 + 450, 260), 2, 2)
        cv2.polylines(cam, [np.array(area2, np.int32)], True, (0, 255, 0), 2)

    elif counter1 < counter2:
        cvzone.putTextRect(cam, f'1', (50 + 450, 260), 2, 2)
        cv2.polylines(cam, [np.array(area1, np.int32)], True, (0, 255, 0), 2)


    cv2.imshow("Smart Counter",cam)
    key=cv2.waitKey(1)
    if key == ord("q"):
        break
