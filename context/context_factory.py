from collections.abc import Callable
from datetime import timedelta
from queue import Queue
import torch
from analysis.activity.multi_person_activity_recon import MultiPersonActivityRecognitionAnalyzer
from analysis.analyzer import BaseAnalyzer
from analysis.analyzer_with_cache_aside import CacheAsideAnalyzer
from classification.activity.suspicious_activity_classifier import SuspiciousActivityClassifier
from classification.behavior.graph_lstm import GraphBasedLSTMClassifier
from classification.classifier import Classifier
from analysis.types import AnalysisType
from classification.people_presence.simple_presence_classifier import SimplePresenceClassifier
from analysis.object_detection.detector import ObjectDetector
from analysis.pose_detection.pose_detector import PoseDetector
from util.device import get_device


class ContextFactory:

    @staticmethod
    def create_analyzers(arg: str) -> list[Callable[[], BaseAnalyzer]]:
        match arg:
            case "behaviour":
                cache_queue = Queue(maxsize=1)
                # FIXME: I don't like the current implementation with the cache-aside analyser
                # TODO: consider the kafka semantics
                return [lambda: CacheAsideAnalyzer(cache_queue, AnalysisType.PersonDetection, ObjectDetector()),
                        lambda: PoseDetector(),
                        lambda: MultiPersonActivityRecognitionAnalyzer(
                            CacheAsideAnalyzer(cache_queue, AnalysisType.PersonDetection), 24, timedelta(seconds=2),12)]
            case "activity":
                return [lambda: MultiPersonActivityRecognitionAnalyzer(ObjectDetector(), 24, timedelta(seconds=2), 12)]
            case "presence":
                return [lambda: ObjectDetector()]

    @staticmethod
    def create_classifier(arg: str, dimensions: tuple[float, float]) -> Callable[[], Classifier]:
        match arg:
            case "behaviour":
                def factory():
                    c = GraphBasedLSTMClassifier(node_features=5, window_size=48, window_step=12,
                                                 dimensions=dimensions).to(get_device())
                    # only on CUDA for the time being due to: https://github.com/pytorch/pytorch/issues/125254
                    c.compile() if torch.cuda.is_available() else None
                    return c

                return factory
            case "activity":
                return lambda: SuspiciousActivityClassifier()
            case "presence":
                return lambda: SimplePresenceClassifier()
