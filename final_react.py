import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import pandas as pd
import numpy as np
import cvzone

def object_tracking(areaData,productionHouse,ip,area):
    stream_data=areaData['streamLink']
    area1=areaData['area1']
    area2=areaData['area2']
    
    model = YOLO("yolov8x.pt")
    cap = cv2.VideoCapture(stream_data)
    
    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")
    
    count = 0
    going_out = {}
    going_in = {}
    cnt1 = []
    cnt2 = []

    while cap.isOpened():
        success, frame = cap.read()
        frame = cv2.resize(frame, (1020, 500))
        
        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=0.4, tracker="botsort.yaml")
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")

            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                id = int(row[4])
                d = int(row[5])

                frame = results[0].plot()
                c = class_list[d]

                if 'person' in c:
                    xc = int(x1 + x2) // 2
                    yc = int(y1 + y2) // 2
                    result = cv2.pointPolygonTest(np.array(area2, np.int32), ((xc, yc)), False)

                    if result >= 0:
                        going_out[id] = (xc, yc)

                    if id in going_out:
                        result1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((xc, yc)), False)
                        if result1 >= 0:
                            cv2.circle(frame, (xc, yc), 7, (255, 0, 255), -1)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                            if cnt1.count(id) == 0:
                                cnt1.append(id)

                    result2 = cv2.pointPolygonTest(np.array(area1, np.int32), ((xc, yc)), False)
                    if result2 >= 0:
                        going_in[id] = (xc, yc)

                    if id in going_in:
                        result3 = cv2.pointPolygonTest(np.array(area2, np.int32), ((xc, yc)), False)
                        if result3 >= 0:
                            cv2.circle(frame, (xc, yc), 7, (255, 0, 255), -1)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                            if cnt2.count(id) == 0:
                                cnt2.append(id)

            # Visualize the results on the frame
            out_c = len(cnt1)
            in_c = len(cnt2)

            cvzone.putTextRect(frame, f'exit:{out_c}', (50, 60), 2, 2)
            cvzone.putTextRect(frame, f'entry:{in_c}', (50, 160), 2, 2)
            cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
            cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function
