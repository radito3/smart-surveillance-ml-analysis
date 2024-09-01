from datetime import timedelta
import concurrent.futures as cf
from math import floor, ceil

import cv2

from .resnet18_3d import ActivityRecognitionAnalyzer
from analysis.analyzer import BaseAnalyzer
from ..types import AnalysisType


class MultiPersonActivityRecognitionAnalyzer(ActivityRecognitionAnalyzer):

    def __init__(self, people_detector: BaseAnalyzer, fps: int, window_size: timedelta, window_step: int):
        assert people_detector.analysis_type() == AnalysisType.PersonDetection, "Analyzer must be PersonDetection"
        super().__init__(fps, window_size, window_step)
        self._people_detector = people_detector

    def buffer_hook(self, frame: cv2.typing.MatLike) -> any:
        yolo_results = self._people_detector.analyze(frame)
        # extract the sub-region for each person
        return [(track_id, self._extract_frame_subregion(frame, bbox)) for bbox, track_id, _ in yolo_results]

    @staticmethod
    def _extract_frame_subregion(frame, bbox) -> cv2.typing.MatLike:
        xmin, ymin, xmax, ymax = bbox
        return frame[floor(ymin):ceil(ymax), floor(xmin):ceil(xmax)]

    def analyze_video_window(self, window: list[list[tuple[any, cv2.typing.MatLike]]]) -> list[any]:
        futures = []

        sub_regions_windows_per_tracking_id = {}
        for people_sub_regions in window:
            for track_id, sub_region in people_sub_regions:
                if track_id.item() in sub_regions_windows_per_tracking_id:
                    sub_regions_windows_per_tracking_id[track_id.item()].append(sub_region)
                else:
                    sub_regions_windows_per_tracking_id[track_id.item()] = [sub_region]

        # Python 3.13 (released 1 Oct 2024) introduces experimental support for free-threading mode (PEP 703)
        # which solves the GIL bottleneck for CPU-bound multithreaded tasks
        executor = cf.ThreadPoolExecutor(max_workers=10)

        for track_id, sub_region_window in sub_regions_windows_per_tracking_id.items():
            # use submit instead of map to preserve order of tracking IDs
            future = executor.submit(super().analyze_video_window, sub_region_window)
            futures.append((track_id, future))

        results = [future.result() for _, future in futures]
        executor.shutdown(wait=False, cancel_futures=True)
        return results
