# import torch
from ultralytics import YOLO
import cv2
import time
import numpy as np
import helpers

### model
model = YOLO("yolov8s")
video = cv2.VideoCapture("basketball.mp4")

### first frame
_, first_frame = video.read()
first_frame = cv2.resize(first_frame, (500, 500))
prevgray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

### create mask for optical flow
mask = np.zeros_like(first_frame)
mask[..., 1] = 255 #set image saturation to maximum



while True:
    ret, frame = video.read()
    if not ret:
        break
    frame = cv2.resize(frame, (500, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame2 = frame.copy()

    ### start
    start = time.time()

    ### find optic flow (angle, motion)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 6, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[..., 1])

    ### detect objects in a frame
    res = model.predict(frame)
    detections = res[0].boxes.data
    
    for detection in detections:
        if detection[-2] >= 0.5:
            x1, y1, x2, y2 = detection[:-2]
            # x, y, w, h = x1, y1, (x2-x1), (y2-y1)
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            area = ((x2-x1) * (y2-y1))/10
            ct = 0
            for i in range(int(x1), int(x2), 10):
                for j in range(int(y1), int(y2), 10):
                    mag, ang = flow[j, i]
                    if abs(mag) >= 1:
                        ct += 1

            print(f"area={area} flow-area={ct} pct={area*0.05} ok={ct >= int(area * 0.5 or (ct >= 50))}")
            if (ct >= int(area * 0.5)) or (ct >= 20):
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    ### end
    end = time.time()
    frame_rate = 1/(end - start)

    helpers.detect_flow(frame2, flow)

    print(f"{frame_rate:.2}")
    cv2.imshow("Video", frame)
    cv2.imshow("Video2", frame2)

    ### set prev frame
    prevgray = gray
    if cv2.waitKey(10) == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()


