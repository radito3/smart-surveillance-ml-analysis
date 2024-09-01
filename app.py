import cv2
import multiprocessing as mp
import multiprocessing.connection as cn
import os
import signal
from typing import Any
from datetime import timedelta
import numpy as np
import torch

from analysis.activity.multi_person_activity_recon import MultiPersonActivityRecognitionAnalyzer
from analysis.activity.resnet18_3d import ActivityRecognitionAnalyzer
from analysis.analyzer import BaseAnalyzer
from analysis.analyzer_with_cache_aside import CacheAsideAnalyzer
from classification.behavior.graph_lstm import GraphBasedLSTMClassifier
from classification.classifier import Classifier
from analysis.types import AnalysisType
from analysis.human_object_interaction.interaction import HumanObjectInteractionAnalyzer
from classification.people_presence.simple_presence_classifier import SimplePresenceClassifier
from notifications.notification_delegate import send_notification
from analysis.object_detection.detector import ObjectDetector
from analysis.optical_flow.optical_flow import OpticalFlowAnalyzer
from analysis.pose_detection.pose_detector import PoseDetector
from analysis.temporal_difference.temporal_difference import TemporalDifferenceAnalyzer
from util.connection_iterator import ConnIterator


def analyzer_wrapper(analyzer: BaseAnalyzer, frame_src: cn.Connection, sink: cn.Connection, sink_lock: mp.Lock):
    for frame in ConnIterator[cv2.typing.MatLike](frame_src):
        vector: list[any] = analyzer.analyze(frame)

        sink_lock.acquire()
        try:
            sink.send((analyzer.analysis_type(), vector))
        finally:
            sink_lock.release()


def sink(classifier: Classifier, sink_conn: cn.Connection) -> None:
    for dtype, data in ConnIterator[tuple[AnalysisType, Any]](sink_conn):
        if classifier.classify_as_suspicious(dtype, data):
            send_notification('localhost:50051')


if __name__ == '__main__':
    os.environ["KERAS_BACKEND"] = "tensorflow"
    # TCP is the underlying transport because UDP can't pass through NAT (at least, according to MediaMTX)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    RTSP_URL = "video.MOV"  # "rtsp://user:pass@192.168.0.189:554/h264Preview_01_main"  # stdin arg
    video_source = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    assert video_source.isOpened(), "Error opening video stream"

    def close_video_capture(signum, frame):
        global video_source
        video_source.release()

    signal.signal(signal.SIGINT, close_video_capture)
    signal.signal(signal.SIGTERM, close_video_capture)

    w, h, fps = (int(video_source.get(x)) for x in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS])
    print(f"width: {w}, height: {h}, fps: {fps}")  # debug only

    # TODO: decide which analysis components to use
    cacheablePeopleDetector = CacheAsideAnalyzer(ObjectDetector(), cache_life=1)
    analyzers: list[BaseAnalyzer] = [cacheablePeopleDetector, PoseDetector(),
                                     MultiPersonActivityRecognitionAnalyzer(cacheablePeopleDetector, fps,
                                                                            timedelta(seconds=1.5), 20)]

    pipes: list[tuple[cn.Connection, cn.Connection]] = [mp.Pipe(duplex=False) for _ in analyzers]
    sink_receiver, sink_sender = mp.Pipe(duplex=False)

    sink_lock: mp.Lock = mp.Lock()
    processes: list[mp.Process] = [mp.Process(target=analyzer_wrapper, args=(analyzer, src, sink_sender, sink_lock,))
                                   for analyzer, (src, _) in zip(analyzers, pipes)]
    [process.start() for process in processes]

    # TODO: decide which classifier to use
    classifier = GraphBasedLSTMClassifier(node_features=5, hidden_dim=16, window_size=int(fps*1.5), window_step=20)
    classifier.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    classifier.eval()

    classifier_process = mp.Process(target=sink, args=(classifier, sink_receiver,))
    classifier_process.start()

    read_attempts: int = 3
    total_frames = fps*3
    frames = 0

    # consider GPU acceleration, Nuitka compilation and/or other optimizations
    while video_source.isOpened() and frames < total_frames:
        ok, frame = video_source.read()  # network I/O
        if not ok and read_attempts > 0:
            read_attempts -= 1
            continue
        if not ok and read_attempts == 0:
            break
        read_attempts = 3  # guard only non-transitive failures

        [dest.send(frame) for _, dest in pipes]
        frames += 1

    # stop on camera disconnect
    video_source.release()

    [dest.send(None) for _, dest in pipes]  # send sentinel value
    [process.join() for process in processes]
    sink_sender.send(None)
    classifier_process.join()
