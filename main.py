import os
import cv2
import numpy as np
import supervision as sv

from tqdm import tqdm
from ultralytics import YOLO

SOURCE_VIDEO_PATH = ''  # add your video path
TARGET_VIDEO_PATH = ''  # add your video path
MODEL = ''  # add your yolov8 model path

model = YOLO(MODEL)
model.fuse()

polygon = np.array([])  # draw your polygon here

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
byte_tracker = sv.ByteTrack(
    track_thresh=0.25,
    track_buffer=video_info.fps,
    match_thresh=0.8, frame_rate=video_info.fps
)
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
zone = sv.PolygonZone(
    polygon=polygon,
    frame_resolution_wh=video_info.resolution_wh
)
box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=0.5
)
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.white(),
    thickness=1,
    text_thickness=1,
    text_scale=0.5
)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
vid_writer = cv2.VideoWriter(
    TARGET_VIDEO_PATH,
    fourcc,
    video_info.fps,
    (video_info.width, video_info.height)
)
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

for frame_num in tqdm(range(video_info.total_frames), desc="Rendering videos with Bounding Box: "):
    ret, frame = cap.read()
    if not ret:
        break

    results = model(
        frame,
        classes=0,
        verbose=False,
        device=''
    )[0]  # set device as yours
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[zone.trigger(detections)]
    detections = byte_tracker.update_with_detections(detections)

    labels = [
        f"{confidence:0.2f}"
        for confidence
        in detections.confidence
    ]

    num_boxes = len(detections)
    count_text = f"Total People: {num_boxes}"

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections,
        labels=labels
    )

    annotated_frame = cv2.putText(
        annotated_frame,
        count_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    vid_writer.write(annotated_frame)

cap.release()
vid_writer.release()
