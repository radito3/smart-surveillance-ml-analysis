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
        AggregateConsumer.__init__(self, broker, ['pose_detection_results', 'video_source'])
        self._num_frames: int = int(fps * window_size.total_seconds())
        self._buffer: list[list[tuple[any, cv2.typing.MatLike]]] = []
        self._window_step: int = window_step
        self._activity_analyzer = None
        self.executor = cf.ThreadPoolExecutor(max_workers=10, thread_name_prefix='activity-analyzer-worker')

    def get_name(self) -> str:
        return 'activity-recognition-app'

    def init(self):
        self._activity_analyzer = ActivityRecognitionAnalyzer()

    def process_message(self, message: dict[str, any]):
        if len(self._buffer) >= self._num_frames:
            result = self.analyze_video_window(self._buffer)
            self.produce_value('activity_detection_results', result)
            self._buffer = self._buffer[self._window_step:]

        frame = message['video_source']
        yolo_results = message['pose_detection_results']
        self._buffer.append(self.extract_sub_regions_for_people(frame, yolo_results))

    def cleanup(self):
        self.produce_value('activity_detection_results', None)
        self.executor.shutdown(wait=False, cancel_futures=True)

    def extract_sub_regions_for_people(self, frame: cv2.typing.MatLike, yolo_results) -> list[tuple[any, cv2.typing.MatLike]]:
        return [(track_id, self._extract_frame_subregion(frame, bbox)) for bbox, track_id, _ in yolo_results]

    @staticmethod
    def _extract_frame_subregion(frame, bbox) -> cv2.typing.MatLike:
        xmin, ymin, xmax, ymax = bbox
        return frame[floor(ymin):ceil(ymax), floor(xmin):ceil(xmax)]

    def analyze_video_window(self, window: list[list[tuple[any, cv2.typing.MatLike]]]) -> list[tuple[any, int]]:
        futures = []

        sub_regions_windows_per_tracking_id = {}
        for people_sub_regions in window:
            for track_id, sub_region in people_sub_regions:
                if track_id.item() in sub_regions_windows_per_tracking_id:
                    sub_regions_windows_per_tracking_id[track_id.item()].append(sub_region)
                else:
                    sub_regions_windows_per_tracking_id[track_id.item()] = [sub_region]

        for track_id, sub_region_window in sub_regions_windows_per_tracking_id.items():
            # use submit instead of map to preserve order of tracking IDs
            future = self.executor.submit(self._activity_analyzer.predict_activity, sub_region_window)
            futures.append((track_id, future))

        return [(track_id, future.result()) for track_id, future in futures]
