import cv2
from threading import Thread
from queue import Queue
import os
import time
import signal
from typing import Any
from datetime import timedelta
import torch

from analysis.activity.multi_person_activity_recon import MultiPersonActivityRecognitionAnalyzer
from analysis.analyzer import BaseAnalyzer
from analysis.analyzer_with_cache_aside import CacheAsideAnalyzer
from classification.behavior.graph_lstm import GraphBasedLSTMClassifier
from classification.classifier import Classifier
from analysis.types import AnalysisType
from classification.people_presence.simple_presence_classifier import SimplePresenceClassifier
from notifications.notification_delegate import send_notification
from analysis.object_detection.detector import ObjectDetector
from analysis.pose_detection.pose_detector import PoseDetector
from util.queue_iterator import QueueIterator
from util.device import get_device


def analyzer_wrapper(analyzer_factory, frame_src: Queue, sink: Queue) -> None:
    analyzer: BaseAnalyzer = analyzer_factory()
    for frame in QueueIterator[cv2.typing.MatLike](frame_src):
        feature_vector = analyzer.analyze(frame)
        if len(feature_vector) > 0:
            sink.put((analyzer.analysis_type(), feature_vector))


def sink(classifier_factory, dims: tuple[float, float], sink_queue: Queue) -> None:
    classifier: Classifier = classifier_factory(dims)
    for dtype, data in QueueIterator[tuple[AnalysisType, Any]](sink_queue):
        with torch.no_grad():
            conf = classifier.classify_as_suspicious(dtype, data)
            if conf != 0:  # experiment with threshold values
                # send_notification('localhost:50051')
                print(conf)


def main(video_url: str, analyzer_factories, classifier_factory) -> None:
    # TCP is the underlying transport because UDP can't pass through NAT (at least, according to MediaMTX)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    # if the kafka semantics are adopted, this can be moved to a dedicated producer class
    # and wrapper producer classes can be added for lower and upper bounding of fps
    video_source = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
    assert video_source.isOpened(), "Error opening video stream"
    width = video_source.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_source.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def close_video_capture(signum, frame):
        video_source.release()

    signal.signal(signal.SIGINT, close_video_capture)
    signal.signal(signal.SIGTERM, close_video_capture)

    queues: list[Queue] = [Queue(maxsize=1) for _ in analyzer_factories]
    sink_queue = Queue(maxsize=1)

    threads: list[Thread] = [Thread(target=analyzer_wrapper, args=(analyzer_factory, queue, sink_queue,))
                             for analyzer_factory, queue in zip(analyzer_factories, queues)]
    [thread.start() for thread in threads]

    classifier_thread = Thread(target=sink, args=(classifier_factory, (width, height), sink_queue,))
    classifier_thread.start()

    # this introduces an upper bound to frame rate
    # however, if the real-time fps is significantly lower than 24, this could lead to issues with analysis performance
    # one mitigation technique is Frame Duplication/Interpolation, but it can be costly
    # for a 1080p @ 60 fps with h.264 encoding video source, a network throughput of at least 8-12 Mbps is required to
    #  reliably transmit the video stream
    target_fps: int = 24  # to ensure consistency between training and analysis
    target_frame_interval: float = 1./target_fps
    # print(f"target inference speed: {target_frame_interval * 1000}ms")
    prev_timestamp: float = 0
    read_attempts: int = 3

    # debug
    total_frames = video_source.get(cv2.CAP_PROP_FPS) * 2
    # print(f"total frames: {total_frames}")
    frames = 0

    while video_source.isOpened() and frames < total_frames:
        time_elapsed: float = time.time() - prev_timestamp
        ok, frame = video_source.read()  # network I/O
        if not ok and read_attempts > 0:
            read_attempts -= 1
            continue
        if not ok and read_attempts == 0:
            break
        read_attempts = 3  # guard only non-transitive failures

        # print(f"inference speed: {time_elapsed * 1000}ms")
        if time_elapsed > target_frame_interval:
            prev_timestamp = time.time()
            [queue.put(frame) for queue in queues]
            frames += 1

    # stop on camera disconnect
    video_source.release()

    [queue.put(None) for queue in queues]  # send sentinel value
    [thread.join() for thread in threads]
    sink_queue.put(None)
    classifier_thread.join()


if __name__ == '__main__':
    cache_queue = Queue(maxsize=1)

    # create the analysers in thread-local storage, instead of on the main thread to avoid parameters corruption
    #  and race conditions
    # reference: https://docs.ultralytics.com/guides/yolo-thread-safe-inference/#understanding-python-threading
    def people_detector_factory() -> BaseAnalyzer:
        return CacheAsideAnalyzer(cache_queue, AnalysisType.PersonDetection, ObjectDetector())

    def pose_detector_factory() -> BaseAnalyzer:
        return PoseDetector()

    def activity_recognition_factory() -> BaseAnalyzer:
        # FIXME: I don't like the current implementation with the cache-aside analyser
        # TODO: consider the kafka semantics
        return MultiPersonActivityRecognitionAnalyzer(CacheAsideAnalyzer(cache_queue, AnalysisType.PersonDetection), 24, timedelta(seconds=2), 12)

    def classifier_factory(dims: tuple[float, float]) -> Classifier:
        c = GraphBasedLSTMClassifier(node_features=5, window_size=48, window_step=12, dimensions=dims).to(get_device())
        # only on CUDA for the time being due to: https://github.com/pytorch/pytorch/issues/125254
        c.compile() if torch.cuda.is_available() else None
        return c

    # GPU:  10.90s user 7.52s system 89% cpu 20.645 total
    # GPU with sequential activity recon:  10.65s user 7.28s system 87% cpu 20.565 total
    # GPU with threads:  11.26s user 7.69s system 96% cpu 19.716 total
    # CPU:  211.61s user 116.33s system 853% cpu 38.438 total
    # turns out the processing issues were not because of the GPU but because of race conditions stemming from
    # creating the models in the main thread and moving them to threads
    main("video.MOV", [people_detector_factory, pose_detector_factory, activity_recognition_factory], classifier_factory)
