import logging
from datetime import timedelta

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

from analysis.activity.multi_person_activity_recon import MultiPersonActivityRecognitionAnalyzer
from analysis.object_detection.detector import ObjectDetector
from analysis.pose_detection.pose_detector import PoseDetector
from messaging.message_broker import MessageBroker
from messaging.source.video_source_producer import VideoSourceProducer


def main(video_url: str, analysis_mode: str, notif_webhook: str):
    broker = MessageBroker()
    source = VideoSourceProducer(broker, video_url)
    object_detector = ObjectDetector(broker)
    pose_detector = PoseDetector(broker)
    activity_detector = MultiPersonActivityRecognitionAnalyzer(broker, 24, timedelta(seconds=2), 12)

    broker.create_topic('video_source')
    broker.create_topic('video_dimensions')
    broker.create_topic('...')

    broker.subscribe_to('video_source', object_detector)
    broker.subscribe_to('video_source', pose_detector)

    if not source.init():
        return

    broker.start()
    broker.wait()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        logging.error("Invalid command-line arguments. Required <video_url> <classification_type> <notification_webhook>")
        sys.exit(1)

    video_url_ = sys.argv[1]
    analysis_mode_ = sys.argv[2]
    notification_webhook = sys.argv[3]

    log_level_env = os.environ['LOG_LEVEL']
    mapping = logging.getLevelNamesMapping()
    if log_level_env is not None and len(log_level_env) != 0 and log_level_env in mapping:
        logging.root.setLevel(mapping[log_level_env])
    else:
        logging.root.setLevel(logging.DEBUG)

    main(video_url_, analysis_mode_, notification_webhook)
