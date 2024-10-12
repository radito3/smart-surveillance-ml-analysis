import cv2
from collections.abc import Callable
from threading import Thread
from queue import Queue
import os
import sys
import time
import signal
from typing import Any
import torch

from analysis.analyzer import BaseAnalyzer
from classification.classifier import Classifier
from analysis.types import AnalysisType
from context.context_factory import ContextFactory
from notifications.notification_delegate import send_notification
from util.queue_iterator import QueueIterator


def analyzer_wrapper(analyzer_factory: Callable[[], BaseAnalyzer], frame_src: Queue, sink: Queue) -> None:
    analyzer: BaseAnalyzer = analyzer_factory()
    for frame in QueueIterator[cv2.typing.MatLike](frame_src):
        feature_vector = analyzer.analyze(frame)
        if len(feature_vector) > 0:
            sink.put((analyzer.analysis_type(), feature_vector))


def sink(classifier_factory: Callable[[], Classifier], sink_queue: Queue) -> None:
    classifier: Classifier = classifier_factory()
    for dtype, data in QueueIterator[tuple[AnalysisType, Any]](sink_queue):
        with torch.no_grad():
            conf = classifier.classify_as_suspicious(dtype, data)
            if conf > 0.6:  # experiment with threshold values
                send_notification(os.environ['NOTIFICATION_SERVICE_URL'])


def main(video_url: str, ctx_factory: ContextFactory, ctx_type: str) -> None:
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

    analyzer_factories = ctx_factory.create_analyzers(ctx_type)
    classifier_factory = ctx_factory.create_classifier(ctx_type, (width, height))

    queues: list[Queue] = [Queue(maxsize=1) for _ in analyzer_factories]
    sink_queue = Queue(maxsize=1)

    threads: list[Thread] = [Thread(target=analyzer_wrapper, args=(analyzer_factory, queue, sink_queue,))
                             for analyzer_factory, queue in zip(analyzer_factories, queues)]
    [thread.start() for thread in threads]

    classifier_thread = Thread(target=sink, args=(classifier_factory, sink_queue,))
    classifier_thread.start()

    # this introduces an upper bound to frame rate
    # however, if the real-time fps is significantly lower than 24, this could lead to issues with analysis performance
    # one mitigation technique is Frame Duplication/Interpolation, but it can be costly
    # for a 1080p @ 60 fps with h.264 encoding video source, a network throughput of at least 8-12 Mbps is required to
    #  reliably transmit the video stream
    target_fps: int = 24  # to ensure consistency between training and analysis
    target_frame_interval: float = 1./target_fps
    prev_timestamp: float = 0
    read_attempts: int = 3

    while video_source.isOpened():
        time_elapsed: float = time.time() - prev_timestamp
        ok, frame = video_source.read()  # network I/O
        if not ok and read_attempts > 0:
            read_attempts -= 1
            continue
        if not ok and read_attempts == 0:
            break
        read_attempts = 3  # guard only non-transitive failures

        if time_elapsed > target_frame_interval:
            prev_timestamp = time.time()
            [queue.put(frame) for queue in queues]

    # stop on camera disconnect
    video_source.release()

    [queue.put(None) for queue in queues]  # send sentinel value
    [thread.join() for thread in threads]
    sink_queue.put(None)
    classifier_thread.join()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Invalid command-line arguments")
        sys.exit(1)

    video_url_ = sys.argv[1]
    ctx_type_ = sys.argv[2]

    main(video_url_, ContextFactory(), ctx_type_)
