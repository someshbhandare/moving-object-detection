import cv2

def detect_flow(frame, flow):
    ct = 0
    for y in range(0, frame.shape[0], 10):
        for x in range(0, frame.shape[1], 10):
            dx, dy = flow[y, x]
            # Draw a bounding box around the optic flow vector
            if abs(dx) < 50:
                continue
            ct += 1
            cv2.rectangle(frame, (x, y), (x + int(dx), y + int(dy)), (0, 255, 0), 2)
    # print(ct)

def segment_frame(frame, magnitude):
    # 
    min_magnitude = 10
    min_cluster_size = 200

    motion_mask = magnitude > min_magnitude
    print(motion_mask)

    # apply motion mask to segmented frame
    segmented_frame = frame.copy()
    segmented_frame[motion_mask] = [0, 0, 255]

    return segmented_frame