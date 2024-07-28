import cv2
import os
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
# even though this is a super minimal and lightweight library, it is from an author of the MMPose project
from rtmlib import draw_skeleton, RTMO
# import anonymization.face_anon_filter as anon

model = YOLO(os.environ["YOLO_MODEL"])
names = model.model.names

# TODO: maybe detect face and hand features separately?
body_pose_detector = RTMO(os.environ["RTMO_MODEL_URL"])

video_path = "output.mov"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS])
print(f"width: {w}, height: {h}, fps: {fps}")

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
desired_frames = 10
end_frame = int(fps * desired_frames)
frame_count = 0

result = cv2.VideoWriter("object_tracking.avi",
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps,
                         (w, h))

# anonymizer = anon.FaceAnonymizer()

while cap.isOpened() and frame_count <= end_frame:
    prev_frame = None
    diff = None
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame,
                            verbose=False,
                            classes=[0],  # match only people
                            conf=0.5)  # confidence cut-off threshold
    # TODO: cut out the part of the frame within the bounding boxes to lessen the overhead of feature extraction
    # find a way to parallelize multi-person detection - might not be needed with RTMO
    boxes = results[0].boxes.xyxy.cpu()
    # print(boxes)

    if results[0].boxes.id is not None:
        # top-level dimension - person N
        # 2nd level - keypoints, scores
        # TODO: filter out keypoints with a lower than 50% score
        keypoints, scores = body_pose_detector(frame)

        # only for debugging
        frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)

        # Extract prediction results
        clss = results[0].boxes.cls.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        # confs = results[0].boxes.conf.float().cpu().tolist()

        # frame = anonymizer(frame, 'pixelate')

        annotator = Annotator(frame, line_width=2)

        for box, cls, track_id in zip(boxes, clss, track_ids):
            annotator.box_label(box, color=colors(int(cls), True), label=f"{names[int(cls)]} {track_id}")

    result.write(frame)
    frame_count += 1

result.release()
cap.release()
# only if displaying any windows
# cv2.destroyAllWindows()
