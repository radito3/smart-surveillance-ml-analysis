import logging
import re
import os
from datetime import timedelta

from analysis.activity.multi_person_activity_recon import MultiPersonActivityRecognitionAnalyzer, SubRegionExtractor
from analysis.human_object_interaction.interaction import HumanObjectInteractionAnalyzer
from classification.activity.suspicious_activity_classifier import SuspiciousActivityClassifier
from classification.behavior.graph_lstm import CompositeBehaviouralClassifier, GraphBasedLSTMClassifier, DimensionsSetter
from analysis.object_detection.object_detector import ObjectDetector
from analysis.pose_detection.pose_detector import PoseDetector
from messaging.message_broker import MessageBroker
from messaging.stream import StreamsBuilder, Stream
from messaging.sink.training_sink import TrainingSink
from notifications.notification_delegate import send_notification


class TopologyBuilder:

    def build_topology_for(self, mode: str, broker: MessageBroker, notification_service_url: str):
        logging.debug(f'Creating streams topology for analysis mode: {mode}')
        match mode:
            case 'behaviour':
                return self.build_behaviour_topology(broker, notification_service_url)
            case 'activity':
                return self.build_activity_topology(broker, notification_service_url)
            case 'presence':
                return self.build_presence_topology(broker, notification_service_url)
            case _:
                raise ValueError(f'Unsupported analysis mode: {mode}')

    @staticmethod
    def build_behaviour_topology(broker: MessageBroker, notification_service_url: str) -> list[Stream]:
        fps = 24  # might be a parameter?
        window_duration = timedelta(seconds=2)
        window_size = fps * window_duration.total_seconds()
        window_step = fps // 2

        classifier = CompositeBehaviouralClassifier(node_features=12)

        probability_threshold: float = 0.6  # default threshold
        if 'SINK_PROBABILITY_THRESHOLD' in os.environ:
            threshold = os.environ['SINK_PROBABILITY_THRESHOLD']
            if bool(re.match(r'^[01]\.\d*$', threshold)):
                probability_threshold: float = float(threshold)

        topics = ['video_source', 'video_dimensions', 'object_detection_results', 'pose_detection_results',
                  'pose_detection_results_batched', 'activity_detection_results', 'hoi_results',
                  'hoi_results_batched']
        for topic in topics:
            logging.debug(f'Creating topic {topic}')
            broker.create_topic(topic)

        builder = StreamsBuilder(broker)

        builder.stream('video_source') \
            .named('pose-detection-app') \
            .process(PoseDetector()) \
            .to('pose_detection_results')

        builder.stream('video_source') \
            .named('object-detection-app') \
            .process(ObjectDetector()) \
            .to('object_detection_results')

        builder.stream('video_dimensions') \
            .named('dimensions-setter-app') \
            .for_each(DimensionsSetter(classifier).process)

        builder.stream('video_source', 'pose_detection_results') \
            .named('activity-recognition-app') \
            .process(SubRegionExtractor()) \
            .window(size=window_size, step=window_step) \
            .process(MultiPersonActivityRecognitionAnalyzer()) \
            .to('activity_detection_results')

        builder.stream('object_detection_results', 'pose_detection_results') \
            .named('human-object-interaction-app') \
            .process(HumanObjectInteractionAnalyzer()) \
            .to('hoi_results')

        builder.stream('pose_detection_results') \
            .named('pose-detection-results-batcher') \
            .window(size=window_size, step=window_step) \
            .to('pose_detection_results_batched')

        builder.stream('hoi_results') \
            .named('hoi-results-batcher') \
            .window(size=window_size, step=window_step) \
            .to('hoi_results_batched')

        builder.stream('pose_detection_results_batched', 'activity_detection_results', 'hoi_results_batched') \
            .named('graph-lstm-classifier-app') \
            .process(classifier) \
            .filter(lambda probability: probability > probability_threshold) \
            .for_each(lambda msg: send_notification(notification_service_url))

        return builder.build()

    @staticmethod
    def build_activity_topology(broker: MessageBroker, notification_service_url: str) -> list[Stream]:
        fps = 24  # might be a parameter?
        window_duration = timedelta(seconds=2)
        window_size = fps * window_duration.total_seconds()
        window_step = fps // 2

        topics = ['video_source', 'video_dimensions', 'pose_detection_results', 'activity_detection_results']
        for topic in topics:
            logging.debug(f'Creating topic {topic}')
            broker.create_topic(topic)

        builder = StreamsBuilder(broker)

        builder.stream('video_source') \
            .named('pose-detection-app') \
            .process(PoseDetector()) \
            .to('pose_detection_results')

        builder.stream('video_source', 'pose_detection_results') \
            .named('activity-recognition-app') \
            .process(SubRegionExtractor()) \
            .window(size=window_size, step=window_step) \
            .process(MultiPersonActivityRecognitionAnalyzer()) \
            .to('activity_detection_results')

        builder.stream('activity_detection_results') \
            .named('suspicious-activity-classifier-app') \
            .process(SuspiciousActivityClassifier()) \
            .for_each(lambda msg: send_notification(notification_service_url))

        return builder.build()

    @staticmethod
    def build_presence_topology(broker: MessageBroker, notification_service_url: str) -> list[Stream]:
        topics = ['video_source', 'video_dimensions']
        for topic in topics:
            logging.debug(f'Creating topic {topic}')
            broker.create_topic(topic)

        builder = StreamsBuilder(broker)

        builder.stream('video_source') \
            .named('simple-presence-classification-app') \
            .process(PoseDetector()) \
            .barrier(3, lambda results: len(results) > 0) \
            .for_each(lambda msg: send_notification(notification_service_url))

        return builder.build()

    @staticmethod
    def build_training_topology(broker: MessageBroker, fps: int, model: GraphBasedLSTMClassifier) -> tuple[list[Stream], TrainingSink]:
        window_size: int = 2 * fps
        window_step: int = fps // 2

        classifier = CompositeBehaviouralClassifier(node_features=12)
        classifier.inject_model(model)

        sink = TrainingSink()

        topics = ['video_source', 'video_dimensions', 'object_detection_results', 'pose_detection_results',
                  'pose_detection_results_batched', 'activity_detection_results', 'hoi_results',
                  'hoi_results_batched']
        for topic in topics:
            logging.debug(f'Creating topic {topic}')
            broker.create_topic(topic)

        builder = StreamsBuilder(broker)

        builder.stream('video_source') \
            .named('pose-detection-app') \
            .process(PoseDetector()) \
            .to('pose_detection_results')

        builder.stream('video_source') \
            .named('object-detection-app') \
            .process(ObjectDetector()) \
            .to('object_detection_results')

        builder.stream('video_dimensions') \
            .named('dimensions-setter-app') \
            .for_each(DimensionsSetter(classifier).process)

        builder.stream('video_source', 'pose_detection_results') \
            .named('activity-recognition-app') \
            .process(SubRegionExtractor()) \
            .window(size=window_size, step=window_step) \
            .process(MultiPersonActivityRecognitionAnalyzer()) \
            .to('activity_detection_results')

        builder.stream('object_detection_results', 'pose_detection_results') \
            .named('human-object-interaction-app') \
            .process(HumanObjectInteractionAnalyzer()) \
            .to('hoi_results')

        builder.stream('pose_detection_results') \
            .named('pose-detection-results-batcher') \
            .window(size=window_size, step=window_step) \
            .to('pose_detection_results_batched')

        builder.stream('hoi_results') \
            .named('hoi-results-batcher') \
            .window(size=window_size, step=window_step) \
            .to('hoi_results_batched')

        builder.stream('pose_detection_results_batched', 'activity_detection_results', 'hoi_results_batched') \
            .named('graph-lstm-classifier-app') \
            .process(classifier) \
            .filter(lambda probability: probability != 0) \
            .for_each(sink.process)

        return builder.build(), sink
