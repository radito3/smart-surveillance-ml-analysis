from datetime import timedelta
import concurrent.futures as cf
from math import floor, ceil
import cv2

from messaging.aggregate_consumer import AggregateConsumer
from messaging.broker_interface import Broker
from messaging.producer import Producer
from .resnet18_3d import ActivityRecognitionAnalyzer


class MultiPersonActivityRecognitionAnalyzer(Producer, AggregateConsumer):

    def __init__(self, broker: Broker, fps: int, window_size: timedelta, window_step: int):
        Producer.__init__(self, broker)
        AggregateConsumer.__init__(self, broker, ['object_detection_results', 'video_source'])
        self._num_frames: int = int(fps * window_size.total_seconds())
        self._buffer: list[list[tuple[any, cv2.typing.MatLike]]] = []
        self._window_step: int = window_step
        self._activity_analyzer = None

    def get_name(self) -> str:
        return 'activity-recognition-app'

    def init(self):
        self._activity_analyzer = ActivityRecognitionAnalyzer()

    def consume_message(self, message: dict[str, any]):
        if len(self._buffer) < self._num_frames:
            frame = message['video_source']
            yolo_results = message['object_detection_results']
            self._buffer.append(self.extract_sub_regions_for_people(frame, yolo_results))
            return
        result = self.analyze_video_window(self._buffer)
        self._buffer = self._buffer[self._window_step:]
        self.produce_value('activity_detection_results', result)

    def cleanup(self):
        self.produce_value('activity_detection_results', None)

    def extract_sub_regions_for_people(self, frame: cv2.typing.MatLike, yolo_results) -> list[tuple[any, cv2.typing.MatLike]]:
        return [(track_id, self._extract_frame_subregion(frame, bbox)) for bbox, track_id, cls, _ in yolo_results if cls == 0]

    @staticmethod
    def _extract_frame_subregion(frame, bbox) -> cv2.typing.MatLike:
        xmin, ymin, xmax, ymax = bbox
        return frame[floor(ymin):ceil(ymax), floor(xmin):ceil(xmax)]

    def analyze_video_window(self, window: list[list[tuple[any, cv2.typing.MatLike]]]) -> list[int]:
        futures = []

        sub_regions_windows_per_tracking_id = {}
        for people_sub_regions in window:
            for track_id, sub_region in people_sub_regions:
                if track_id.item() in sub_regions_windows_per_tracking_id:
                    sub_regions_windows_per_tracking_id[track_id.item()].append(sub_region)
                else:
                    sub_regions_windows_per_tracking_id[track_id.item()] = [sub_region]

        # Python 3.13 introduces experimental support for free-threading mode (PEP 703)
        # which solves the GIL bottleneck for CPU-bound multithreaded tasks
        executor = cf.ThreadPoolExecutor(max_workers=10)

        for track_id, sub_region_window in sub_regions_windows_per_tracking_id.items():
            # use submit instead of map to preserve order of tracking IDs
            future = executor.submit(self._activity_analyzer.analyze_video_window, sub_region_window)
            futures.append((track_id, future))

        results = [future.result() for _, future in futures]
        executor.shutdown(wait=False, cancel_futures=True)
        return results
