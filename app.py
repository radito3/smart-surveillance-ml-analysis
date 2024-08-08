import cv2
import multiprocessing as mp
import multiprocessing.connection as cn
import os
from datetime import timedelta

from activity.activity_recon import ActivityRecognitionAnalyzer
from analysis.analyzer import BaseAnalyzer
from analysis_process import analyzer_wrapper
from human_object_interaction.interaction import HumanObjectInteractionAnalyzer
from object_detection.detector import ObjectDetector
from optical_flow.optical_flow import OpticalFlowAnalyzer
from pose_detection.pose_detector import PoseDetector
from temporal_difference.temporal_difference import TemporalDifferenceAnalyzer


def sink(sink_conn: cn.Connection) -> None:
    while True:
        obj = sink_conn.recv()
        if obj is None:
            return

        if len(obj['data']) != 0:
            print(obj)


if __name__ == '__main__':
    os.environ["KERAS_BACKEND"] = "tensorflow"
    # TCP is the underlying transport because UDP can't pass through NAT (at least, according to MediaMTX)
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    RTSP_URL = "rtsp://user:pass@192.168.0.189:554/h264Preview_01_main"  # stdin arg
    video_source = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    assert video_source.isOpened(), "Error opening video stream"

    w, h, fps = (int(video_source.get(x)) for x in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS])
    print(f"width: {w}, height: {h}, fps: {fps}")  # debug only

    frame_receiver, frame_sender = mp.Pipe(duplex=False)
    sink_receiver, sink_sender = mp.Pipe(duplex=False)

    # TODO: decide which analysis components to use
    analyzers: list[BaseAnalyzer] = [HumanObjectInteractionAnalyzer(ObjectDetector()), PoseDetector(),
                                     OpticalFlowAnalyzer(), TemporalDifferenceAnalyzer(),
                                     ActivityRecognitionAnalyzer(fps, timedelta(seconds=3))]

    pipes: list[tuple[cn.Connection, cn.Connection]] = [mp.Pipe(duplex=False) for _ in analyzers]

    sink_lock: mp.Lock = mp.Lock()
    processes: list[mp.Process] = [mp.Process(target=analyzer_wrapper, args=(analyzer, src, sink_sender, sink_lock,))
                                   for analyzer, (src, _) in zip(analyzers, pipes)]
    [process.start() for process in processes]

    classifier = mp.Process(target=sink, args=(sink_receiver,))
    classifier.start()

    read_attempts: int = 3

    # through some experiments, it takes a little over 5 seconds for YOLO to process 24 frames
    # this is with pure python, running on CPU
    # for a real-time scenario, this would really lag behind
    # GPU acceleration, Nuitka compilation and/or other optimizations should be considered
    while video_source.isOpened():
        ok, frame = video_source.read()  # network I/O
        if not ok and read_attempts > 0:
            read_attempts -= 1
            continue
        if not ok and read_attempts == 0:
            break
        read_attempts = 3  # guard only non-transitive failures

        [dest.send(frame) for _, dest in pipes]

    # stop on camera disconnect
    video_source.release()

    [dest.send(None) for _, dest in pipes]  # send sentinel value
    [process.join() for process in processes]
    sink_sender.send(None)

    classifier.join()
