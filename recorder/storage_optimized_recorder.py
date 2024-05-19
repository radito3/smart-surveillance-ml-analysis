import cv2
import numpy as np


def compare_frames(frame1, frame2, threshold=30):
    """Compare two frames and return True if they are different."""
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    non_zero_count = np.count_nonzero(thresh)
    return non_zero_count > 0


"""
this will be a pre-processing filter when recording video streams to GCS/k8s Storage class/persistent volume
and there will be a cron job that will periodically merge file parts (and maybe compress them)
"""
def process_video(input_path, output_path, idle_time=300, low_fps=1, high_fps=24):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, high_fps, (width, height))

    frame_count = 0
    idle_frames = 0
    current_fps = high_fps

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read the video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if compare_frames(prev_frame, frame):
            idle_frames = 0
            if current_fps != high_fps:
                current_fps = high_fps
                out = cv2.VideoWriter(output_path, fourcc, high_fps, (width, height))
        else:
            idle_frames += 1
            if idle_frames >= idle_time * current_fps:
                current_fps = low_fps
                out = cv2.VideoWriter(output_path, fourcc, low_fps, (width, height))

        out.write(frame)
        prev_frame = frame
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
input_path = 'input.mp4'
output_path = 'vid<date>_<time>.<buffer_part>.avi'
process_video(input_path, output_path)
