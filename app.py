import cv2
import multiprocessing as mp
import multiprocessing.connection as cn
import selectors
from typing import cast
import os
import time
import main

main.say_hello_to('ZAWARUDO')

sel = selectors.DefaultSelector()


def sleeper_process(conn: cn.Connection, selector: selectors.BaseSelector, stop_signal_fileno: int):
    while True:
        time.sleep(1)
        events = selector.select(0)
        for key, _ in events:
            if key.fd == stop_signal_fileno:
                return
        conn.send(True)

# events = sel.select()
# for key, _ in events:
#     key.fileobj  # generic object
#     # cast(cn.Connection, selectorKey.fileobj)


# stdin arg: media mtx camera source URL
RTSP_URL = 'rtsp://user:pass@192.168.0.189:554/h264Preview_01_main'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
video_source = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
assert video_source.isOpened(), "Error opening video stream"

w, h, fps = (int(video_source.get(x)) for x in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS])
print(f"width: {w}, height: {h}, fps: {fps}")


stop_signal_receiver, stop_signal_sender = mp.Pipe(duplex=False)

timer_receiver, timer_sender = mp.Pipe(duplex=False)

frame_receiver, frame_sender = mp.Pipe(duplex=False)

sel.register(frame_receiver, selectors.EVENT_READ)
sel.register(timer_receiver, selectors.EVENT_READ)
sel.register(stop_signal_receiver, selectors.EVENT_READ)
# TODO: pass the selector to the processes due to shared memory constraints

sleeper = mp.Process(target=sleeper_process, args=(timer_sender, sel, stop_signal_receiver.fileno(),))
sleeper.start()
analysis_pipeline = mp.Process()
analysis_pipeline.start()
decider = mp.Process()
decider.start()


read_attempts: int = 3
frame_buffer: list[cv2.typing.MatLike] = []  # where should this buffer be?

while video_source.isOpened():
    ok, frame = video_source.read()  # network I/O
    if not ok and read_attempts > 0:
        read_attempts -= 1
        continue
    if not ok and read_attempts == 0:
        break
    read_attempts = 3

    frame_sender.send(frame)

stop_signal_sender.send(True)

# stop on camera disconnect
sleeper.join()
analysis_pipeline.join()
decider.join()
