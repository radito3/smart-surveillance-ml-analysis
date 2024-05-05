import cv2
import numpy as np
from ultralytics import YOLO

# from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

# from collections import defaultdict

# track_history = defaultdict(lambda: [])
model = YOLO("yolov8s.pt")
names = model.model.names

# TODO: to use an OpenCV MediaMTX producer with GStreamer, I need to build OpenCV from source...
# print(cv2.getBuildInformation())

video_path = "output.mov"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

print("video opened")

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

while cap.isOpened() and frame_count <= end_frame:
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, verbose=False, classes=[0], conf=0.5)
        boxes = results[0].boxes.xyxy.cpu()
        # print(boxes)

        if results[0].boxes.id is not None:
            # Extract prediction results
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # confs = results[0].boxes.conf.float().cpu().tolist()

            # Annotator Init
            annotator = Annotator(frame, line_width=2)

            for box, cls, track_id in zip(boxes, clss, track_ids):
                annotator.box_label(box, color=colors(int(cls), True), label=f"{names[int(cls)]} {track_id}")

                # Store tracking history
                # track = track_history[track_id]
                # track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                # if len(track) > 30:
                #     track.pop(0)
                #
                # Plot tracks
                # points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                # cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                # cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

        result.write(frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

result.release()
cap.release()
cv2.destroyAllWindows()
