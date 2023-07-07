import cv2
import numpy as np
import torch
import os
import sys
import json
import time

from tools import generate_keypoint_profile, RunningAverage, circular_distance, count_pixels, plot_path, determine_angle_kps
from sort.sort import Sort
sys.path.append("yolov5")
from models.common import DetectMultiBackend
from utils.plots import Annotator
from utils.general import non_max_suppression, scale_boxes


def remove_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


# remove_files("dataset_test")

num = 0
mapping = {}
distance_threshold = 75
width = 640
height = 640
log = {}
output_json = "fight_data.json"
circle_center = None
circle_radius = 100

mot_tracker = Sort()
kps_tracker = generate_keypoint_profile(cv2.imread("plane_vert.png", 0))

weights = "yolov5/runs/train/exp28/weights/best.pt"
model = DetectMultiBackend(weights, device='cpu')
cap = cv2.VideoCapture("movie.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = 'analyzer_video2.mp4'
fps = 24
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
time_stamp = 0
draw = True

self_vector = None
self_points = []
self_color = (255, 0, 0)
track_color = (0, 0, 0)
counter = 0
while cap.isOpened():
    start_frame = time.time()
    try:
        ret, im0s = cap.read()
        counter += 1
        if counter % 10 != 0: continue
        im0s = cv2.resize(im0s, (width, height))
        transposed = np.transpose(im0s, (2, 0, 1))
        flipped = np.flip(transposed, axis=0)
        flipped = flipped.copy()
        im = torch.from_numpy(flipped).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pred = model(im)
        pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.1)
        start_stage_2 = time.time()
        overlay = im0s.copy()
        im0 = im0s.copy()
        for i, det in enumerate(pred):

            if len(det):

                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                dets = []
                for *xyxy, conf, cls in reversed(det):
                    dets.append([*xyxy, conf])


                track_bbs_ids = np.flip(mot_tracker.update(np.array(dets)), axis=0)

                self_vector = None
                for j in range(track_bbs_ids.shape[0]):
                    num += 1
                    start_j = time.time()

                    xyxy = track_bbs_ids[j, :4]
                    xyxy = [0 if num < 0 else num for num in xyxy]
                    xyxy = [640 if num > 640 else num for num in xyxy]

                    id = track_bbs_ids[j, 4]
                    center = (int((xyxy[0] + xyxy[2]) // 2), int((xyxy[1] + xyxy[3]) // 2))

                    symbol_region = im0s[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    symbol_region_resized = cv2.resize(symbol_region, None, fx=16, fy=16)
                    track_rgb = count_pixels(symbol_region, self_color, track_color)
                    track_bgr = cv2.cvtColor(np.uint8([[track_rgb]]), cv2.COLOR_RGB2BGR)[0][0]
                    track_bgr = (int(track_bgr[0]), int(track_bgr[1]), int(track_bgr[2]))

                    cv2.imwrite(f"dataset_test/preresized{num}.png", symbol_region)
                    cv2.imwrite(f"dataset_test/precanny{num}.png", symbol_region_resized)
                    symbol_region_resized = cv2.imread(f"dataset_test/precanny{num}.png")

                    try:
                        distance, angle = determine_angle_kps(symbol_region_resized, kps_tracker)
                        print(num, angle)
                    except ValueError as e:
                        print(e)
                        continue
                    except np.linalg.LinAlgError as e:
                        print(e)
                        continue
                    if distance > distance_threshold:
                        color = (0, 0, 255)
                        cv2.imwrite(f"failed/preresized{num}.png", symbol_region)
                        cv2.imwrite(f"failed/precanny{num}.png", symbol_region_resized)
                    else:
                        color = (0, 255, 0)

                    if id not in mapping:
                        mapping[id] = RunningAverage(threshold=distance_threshold)
                    mapping[id].update(angle, distance)

                    avg_angle = mapping[id].get_angle() if mapping[id].get_angle() else angle

                    if track_rgb == self_color:
                        self_vector = avg_angle
                        circle_center = center
                        self_points.append(center)
                        if draw:
                            im0 = plot_path(im0, self_points, track_bgr)

                    label = f"{int(id)}-{int(angle)}-{int(distance)}-{num}"

                    if circle_center:

                        if id not in log:
                            log[id] = []
                        displacement_vector = circular_distance(avg_angle, self_vector) if avg_angle and self_vector else None
                        log[id].append([time_stamp, avg_angle, None, displacement_vector])

                    if draw:
                        filled_image = np.zeros_like(im0, dtype=np.uint8)
                        track_bgr = (int(track_bgr[0]), int(track_bgr[1]), int(track_bgr[2]))
                        cv2.putText(im0, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 3)
                        center = [int(xyxy[0] + xyxy[2]) // 2, int(xyxy[1] + xyxy[3]) // 2]
                        length = 100
                        end_point = (
                            int(center[0] + length * np.cos(np.deg2rad(avg_angle - 90))),
                            int(center[1] + length * np.sin(np.deg2rad(avg_angle - 90)))
                        )
                        cv2.line(im0, center, end_point, color, 1)
                    print(f"j time: {time.time() - start_j}")

            alpha = 0.25
            if draw:
                cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0, im0)

            if circle_center:
                cv2.circle(im0, circle_center, circle_radius, (255, 0, 0), 2)

            time_stamp += 1
            out.write(im0)
            cv2.imshow("Image", im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        with open(output_json, "w+") as file:
            json.dump(log, file)
        break
    print(f"stage2 {time.time() - start_stage_2}")
    print(f"Frame {time.time() - start_frame}\n\n")

with open(output_json, "w+") as file:
    json.dump(log, file)
out.release()
cap.release()
cv2.destroyAllWindows()
