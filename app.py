import logging
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


def sink(classifier_factory: Callable[[], Classifier], notif_webhook: str, sink_queue: Queue) -> None:
    classifier: Classifier = classifier_factory()
    for dtype, data in QueueIterator[tuple[AnalysisType, Any]](sink_queue):
        with torch.no_grad():
            conf = classifier.classify_as_suspicious(dtype, data)
            if conf > 0.6:  # experiment with threshold values
                send_notification(notif_webhook)


def main(video_url: str, ctx_factory: ContextFactory, ctx_type: str, notif_webhook: str) -> None:
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

    classifier_thread = Thread(target=sink, args=(classifier_factory, notif_webhook, sink_queue,))
    classifier_thread.start()

    # this introduces an upper bound to frame rate
    # however, if the real-time fps is significantly lower than 24, this could lead to issues with analysis performance
    # one mitigation technique is Frame Duplication/Interpolation, but it can be costly
    # for a lower bound implementation (but it does have frame dropping, so it might not be desirable), see:
    # https://medium.com/@vikas.c20/optimizing-rtsp-video-processing-in-opencv-overcoming-fps-discrepancies-and-buffering-issues-463e204c7b86
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
            logging.debug("Connection issue: Retrying after 2 seconds")
            time.sleep(2)
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
    if len(sys.argv) != 4:
        logging.error("Invalid command-line arguments. Required <video_url> <classification_type> <notification_webhook>")
        sys.exit(1)

    video_url_ = sys.argv[1]
    ctx_type_ = sys.argv[2]
    notification_webhook = sys.argv[3]

    log_level_env = os.environ['LOG_LEVEL']
    mapping = logging.getLevelNamesMapping()
    if log_level_env is not None and len(log_level_env) != 0 and log_level_env in mapping:
        logging.root.setLevel(mapping[log_level_env])
    else:
        logging.root.setLevel(logging.DEBUG)

    main(video_url_, ContextFactory(), ctx_type_, notification_webhook)
