import logging
import os
import time
from threading import Thread

import cv2

from messaging.broker_interface import Broker
from messaging.producer import Producer


class VideoSourceProducer(Producer):

    def __init__(self, broker: Broker, video_url: str, with_upper_fps_limit: bool = True):
        super().__init__(broker)
        self.video_url: str = video_url
        self.video_capture = None
        self.upper_fps_limit: bool = with_upper_fps_limit

    def get_name(self) -> str:
        return 'video-source-producer'

    def init(self):
        # TCP is the underlying transport because UDP can't pass through NAT (at least, according to MediaMTX)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.video_capture = cv2.VideoCapture(self.video_url, cv2.CAP_FFMPEG)
        if not self.video_capture.isOpened():
            # TODO: sanitize the URL, masking any credentials when connecting to a secure endpoint
            raise Exception(f"Could not open video stream for: {self.video_url}")

        width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.produce_value('video_dimensions', (width, height))
        self.produce_value('video_dimensions', None)

        worker = Thread(name=self.get_name() + '-thread', target=self._produce_video_frames, daemon=True)
        worker.start()

    def _safe_write(self, message: any) -> bool:
        try:
            self.produce_value('video_source', message)
            return True
        except StopIteration:
            return False

    def _produce_video_frames(self):
        # this introduces an upper bound to frame rate
        # however, if the real-time fps is significantly lower than 24, this could lead to issues with analysis performance
        # one mitigation technique is Frame Duplication/Interpolation, but it can be costly
        # for a lower bound implementation, see:
        # https://medium.com/@vikas.c20/optimizing-rtsp-video-processing-in-opencv-overcoming-fps-discrepancies-and-buffering-issues-463e204c7b86
        # for a 1080p @ 60 fps with h.264 encoding video source, a network throughput of at least 8-12 Mbps is required to
        #  reliably transmit the video stream
        target_fps: int = 24
        target_frame_interval: float = 1./target_fps
        prev_timestamp: float = 0
        read_attempts: int = 3

        while self.video_capture.isOpened():
            time_elapsed: float = time.time() - prev_timestamp
            ok, frame = self.video_capture.read()  # network I/O
            if not ok and read_attempts > 0:
                read_attempts -= 1
                logging.debug("Connection issue: Retrying after 2 seconds")
                time.sleep(2)
                continue
            if not ok and read_attempts == 0:
                self._safe_write(None)
                break
            read_attempts = 3  # guard only non-transitive failures

            if not self.upper_fps_limit or time_elapsed > target_frame_interval:
                prev_timestamp = time.time()
                # even though we may drop a few frames here and there, that should be acceptable
                # if the source is running at too much fps, this upper limit safeguards us from overloading the system
                if not self._safe_write(frame):
                    break

        # stop on camera disconnect
        self.video_capture.release()
        self._safe_write(None)
