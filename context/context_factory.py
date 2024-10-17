import os
from collections.abc import Callable
from datetime import timedelta
from queue import Queue
import torch
from analysis.activity.multi_person_activity_recon import MultiPersonActivityRecognitionAnalyzer
from classification.activity.suspicious_activity_classifier import SuspiciousActivityClassifier
from classification.behavior.graph_lstm import GraphBasedLSTMClassifier
from classification.people_presence.simple_presence_classifier import SimplePresenceClassifier
from analysis.object_detection.detector import ObjectDetector
from analysis.pose_detection.pose_detector import PoseDetector
from messaging.consumer import Consumer
from messaging.message_broker import MessageBroker
from messaging.producer import Producer
from util.device import get_device


# maybe rename to TopologyBuilder? TopologyFactory?
# maybe have state?
class ContextFactory:

    @staticmethod
    def build_topology(mode: str):
        broker = MessageBroker()
        match mode:
            case "behaviour":
                broker.create_topic('video_source')
                broker.create_topic('...')
            case "activity":
                pass
            case "presence":
                pass
        return broker

    @staticmethod
    def register_producers(broker: MessageBroker, producers: list[Producer]):
        pass

    @staticmethod
    def register_consumers(broker: MessageBroker, consumers: list[Consumer]):
        pass

    # @staticmethod
    # def create_analyzers(arg: str) -> list[Callable[[], BaseAnalyzer]]:
    #     match arg:
    #         case "behaviour":
    #             cache_queue = Queue(maxsize=1)
    #             return [lambda: CacheAsideAnalyzer(cache_queue, AnalysisType.PersonDetection, ObjectDetector()),
    #                     lambda: PoseDetector(),
    #                     lambda: MultiPersonActivityRecognitionAnalyzer(
    #                         CacheAsideAnalyzer(cache_queue, AnalysisType.PersonDetection), 24, timedelta(seconds=2), 12)]
    #         case "activity":
    #             return [lambda: MultiPersonActivityRecognitionAnalyzer(ObjectDetector(), 24, timedelta(seconds=2), 12)]
    #         case "presence":
    #             return [lambda: ObjectDetector()]
    #
    # @staticmethod
    # def create_classifier(arg: str, dimensions: tuple[float, float]) -> Callable[[], Classifier]:
    #     match arg:
    #         case "behaviour":
    #             def factory():
    #                 c = GraphBasedLSTMClassifier(node_features=5, window_size=48, window_step=12,
    #                                              dimensions=dimensions).to(get_device())
    #                 pretrained_weights_path = os.environ['GRAPH_LSTM_WEIGHTS_PATH']
    #                 if pretrained_weights_path is not None and len(pretrained_weights_path) != 0:
    #                     c.load_state_dict(torch.load(pretrained_weights_path, map_location=get_device()))
    #                 # only on CUDA due to: https://github.com/pytorch/pytorch/issues/125254
    #                 c.compile() if torch.cuda.is_available() else None
    #                 return c
    #
    #             return factory
    #         case "activity":
    #             return lambda: SuspiciousActivityClassifier()
    #         case "presence":
    #             return lambda: SimplePresenceClassifier()
