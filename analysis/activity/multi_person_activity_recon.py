import concurrent.futures as cf
from math import floor, ceil
from collections import defaultdict
import cv2

from messaging.processor import MessageProcessor
from .resnet18_3d import ActivityRecognitionAnalyzer


class SubRegionExtractor(MessageProcessor):

    def __init__(self):
        super().__init__()

    def process(self, message: any):
        frame = message['video_source']
        yolo_results = message['pose_detection_results']
        result = [(track_id, self._extract_frame_subregion(frame, bbox)) for bbox, track_id, _ in yolo_results]
        self.next(result)

    @staticmethod
    def _extract_frame_subregion(frame, bbox) -> cv2.typing.MatLike:
        xmin, ymin, xmax, ymax = bbox
        return frame[floor(ymin):ceil(ymax), floor(xmin):ceil(xmax)]


class MultiPersonActivityRecognitionAnalyzer(MessageProcessor):

    def __init__(self):
        super().__init__()
        self.activity_analyzer = None
        self.executor = cf.ThreadPoolExecutor(max_workers=10, thread_name_prefix='activity-analyzer-worker')

    def init(self):
        self.activity_analyzer = ActivityRecognitionAnalyzer()

    def process(self, window: list[list[tuple[any, cv2.typing.MatLike]]]):
        futures = []
        subregions_across_frames_per_tracking_id = self.aggregate_people_subregions_across_frames(window)

        for track_id, sub_region_window in subregions_across_frames_per_tracking_id.items():
            # use submit instead of map to preserve order of tracking IDs
            future = self.executor.submit(self.activity_analyzer.predict_activity, sub_region_window)
            futures.append((track_id, future))

        result = [(track_id, future.result()) for track_id, future in futures]
        self.next(result)

    @staticmethod
    def aggregate_people_subregions_across_frames(window: list[list[tuple[any, cv2.typing.MatLike]]]) -> dict[any, list[cv2.typing.MatLike]]:
        subregions_across_frames = defaultdict(list)

        for people_sub_regions in window:
            for track_id, sub_region in people_sub_regions:
                subregions_across_frames[track_id.item()].append(sub_region)

        return subregions_across_frames

    def cleanup(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
