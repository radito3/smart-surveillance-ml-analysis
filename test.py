import cv2
import os
import numpy as np
from ultralytics import YOLO
# from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors
# even though this is a super minimal and lightweight library, it is from an author of the MMPose project
from rtmlib import draw_skeleton, RTMO
from skimage.feature import hog as histogram_of_oriented_gradients
# import tensorflow.keras.applications.DenseNet201
# from collections import defaultdict
import anonymization.face_anon_filter as anon

# track_history = defaultdict(lambda: [])
model = YOLO(os.environ["YOLO_MODEL"])
names = model.model.names

# TODO: maybe detect face and hand features separately?
body_pose_detector = RTMO(os.environ["RTMO_MODEL_URL"])

# RTSP_URL = 'rtsp://user:pass@192.168.0.189:554/h264Preview_01_main'
# os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
# cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

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

anonymizer = anon.FaceAnonymizer()


def temporal_difference(frame1, frame2):
    # Convert frames to grayscale
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return cv2.absdiff(gray_frame1, gray_frame2)


def compute_optical_flow_feature(frame1, frame2):
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # Calculate optical flow using the Farneback method
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Compute magnitude and angle of the optical flow vectors
    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    # Flatten magnitude and angle arrays into feature vector
    mag_feature = mag.flatten()
    ang_feature = ang.flatten()
    # Concatenate magnitude and angle features
    return np.concatenate((mag_feature, ang_feature))


while cap.isOpened() and frame_count <= end_frame:
    prev_frame = None
    diff = None
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame,
                          persist=True,  # persists the trackers between different calls
                          verbose=False,
                          classes=[0],  # match only people
                          conf=0.5)  # confidence cut-off threshold
    # TODO: cut out the part of the frame withing the bounding boxes to lessen the overhead of feature extraction
    # find a way to parallelize multi-person detection - might not be needed with RTMO
    boxes = results[0].boxes.xyxy.cpu()
    # print(boxes)

    if results[0].boxes.id is not None:
        if prev_frame is not None:
            diff = temporal_difference(prev_frame, frame)

            # should this be here or outside the loop?
            mask = np.zeros_like(prev_frame)

            next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Calculate optical flow using the Farneback method
            flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # Compute the magnitude and angle of the optical flow vectors
            mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
            # Convert angle to Hue
            mask[:, :, 0] = ang * 180 / np.pi / 2
            # Normalize the magnitude
            mask[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # Convert HSV to RGB
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        else:
            # skip processing for the first frame, since we don't have a difference yet
            pass

        prev_frame = frame

        # TODO: maybe calculate this on the difference frames, rather than individual frames
        hst_descriptor = histogram_of_oriented_gradients(frame,
                                                         # (for colored images)
                                                         # indicates which axis of the array corresponds to channels
                                                         channel_axis=-1)
        print(hst_descriptor)

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

        frame = anonymizer(frame, 'pixelate')

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

result.release()
cap.release()
# only if displaying any windows
# cv2.destroyAllWindows()
