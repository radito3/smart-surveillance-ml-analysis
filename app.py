import cv2
import multiprocessing as mp
import multiprocessing.connection as cn
import os
import time
from datetime import timedelta
from analysis_process import analysis


def sleeper_process(conn: cn.Connection, stop_signal: cn.Connection) -> None:
    while True:
        time.sleep(1)
        if stop_signal.poll():
            return
        conn.send(True)


# stdin arg: media mtx camera source URL
RTSP_URL = 'rtsp://user:pass@192.168.0.189:554/h264Preview_01_main'
# TCP is the underlying transport because UDP can't pass through NAT (at least, according to MediaMTX)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
video_source = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
assert video_source.isOpened(), "Error opening video stream"

w, h, fps = (int(video_source.get(x)) for x in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS])
# debug only
print(f"width: {w}, height: {h}, fps: {fps}")


stop_signal_receiver, stop_signal_sender = mp.Pipe(duplex=False)
timer_receiver, timer_sender = mp.Pipe(duplex=False)
frame_receiver, frame_sender = mp.Pipe(duplex=False)
sink_receiver, sink_sender = mp.Pipe(duplex=False)

sleeper = mp.Process(target=sleeper_process, args=(timer_sender, stop_signal_receiver,))
sleeper.start()

# TODO: connect sleep timer with analysis pipeline
analysis_pipeline = mp.Process(target=analysis, args=(frame_receiver, fps, timedelta(seconds=3),
                                                      sink_receiver, stop_signal_receiver,))
analysis_pipeline.start()

# TODO: the sink should collect all feature vectors, flatten them, remove empty values, pad (if necessary) and either
#  pass to the pre-trained LSTM model (trained via a different program and saved as a binary file with pickle
#  serialization) or just fire a notification
decider = mp.Process()
decider.start()


read_attempts: int = 3

while video_source.isOpened():
    ok, frame = video_source.read()  # network I/O
    if not ok and read_attempts > 0:
        read_attempts -= 1
        continue
    if not ok and read_attempts == 0:
        break
    read_attempts = 3  # guard only non-transitive failures

    frame_sender.send(frame)

stop_signal_sender.send(True)
video_source.release()

# stop on camera disconnect
sleeper.join()
analysis_pipeline.join()
decider.join()
