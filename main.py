import argparse
import time

from ultralytics import YOLO
import cv2
import numpy as np

### arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input-video-path", required=True)
parser.add_argument("--output-video-path", default="output.mp4")
args = parser.parse_args()

### input and output path
INPUT_VIDEO_PATH = args.input_video_path
OUTPUT_VIDEO_PATH = args.output_video_path
# OUTPUT_VIDEO_PATH = "output.mp4"
print(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)

### model
model = YOLO("yolov8s")
video = cv2.VideoCapture(INPUT_VIDEO_PATH)

VIDEO_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
VIDEO_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
VIDEO_FPS = video.get(cv2.CAP_PROP_FPS)
# print(f"fps: {VIDEO_FPS}")
output_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (640, 480))


### first frame
_, first_frame = video.read()
first_frame = cv2.resize(first_frame, (640,480))
prevgray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

### create mask for optical flow
mask = np.zeros_like(first_frame)
mask[..., 1] = 255 #set image saturation to maximum

### frame_count
frame_count = 0
ct = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    frame_count += 1
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame2 = frame.copy()

    ### start
    start = time.time()

    ### detect objects in a frame
    res = model.predict(frame, verbose=False, imgsz=320)
    detections = res[0].boxes.data
    class_labels = res[0].names
    
    for detection in detections:
        if detection[-2] >= 0.5:
            x1, y1, x2, y2 = detection[:-2]

            if frame_count%5 == 1:
                ### find optic flow (angle, motion)
                flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 6, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                magnitude, angle = cv2.cartToPolar(flow[...,0], flow[..., 1])

                ### set prev frame
                prevgray = gray

                ct = 0
                # width = int(x2-x1)
                # height = int(y2-y1)
                # area = int(width*height/10)
                for i in range(int(x1), int(x2), 10):
                    for j in range(int(y1), int(y2), 10):
                        mag, ang = flow[j, i]
                        if abs(mag) >= 1:
                            ct += mag

            if (abs(ct) >= 150):
                print("="*40)
                print(f"class_name: {class_labels[int(detection[-1])]}, ct: {ct}")
                print("="*40)
                cv2.rectangle(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame2, class_labels[int(detection[-1])], (int(x1+1), int(y1-5)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

    ### end
    end = time.time()
    frame_rate = 1/(end - start)

    print(f"fps: {frame_rate:.2f}")
    cv2.imshow("Original Video", frame)
    cv2.imshow("Moving Object Detection Video", frame2)

    ### write to output video
    output_video.write(frame2)
    if cv2.waitKey(10) == ord('q'):
        break

print("Output video generated")
video.release()
cv2.destroyAllWindows()


