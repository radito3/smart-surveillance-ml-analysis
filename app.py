import logging
import re
import os
import signal
import sys

from analysis.activity.multi_person_activity_recon import MultiPersonActivityRecognitionAnalyzer, SubRegionExtractor
from analysis.human_object_interaction.interaction import HumanObjectInteractionAnalyzer
from analysis.object_detection.object_detector import ObjectDetector
from analysis.pose_detection.pose_detector import PoseDetector
from classification.activity.suspicious_activity_classifier import SuspiciousActivityClassifier
from classification.behavior.graph_lstm import CompositeBehaviouralClassifier, DimensionsSetter
from messaging.message_broker import MessageBroker
from messaging.streams_builder import StreamsBuilder
from messaging.topology import Topology, KafkaStreams
from messaging.sink.notification_sink import NotificationSink
from messaging.source.video_source_producer import VideoSourceProducer


def setup_logger():
    logging.basicConfig(format="%(threadName)s: %(message)s")
    mapping = logging.getLevelNamesMapping()
    if 'LOG_LEVEL' in os.environ:
        log_level_env = os.environ['LOG_LEVEL']
        if log_level_env in mapping:
            logging.root.setLevel(mapping[log_level_env])
    else:
        logging.root.setLevel(logging.DEBUG)


def build_topology_for(mode: str, broker: MessageBroker) -> Topology:
    logging.debug(f'Creating streams topology for analysis mode: {mode}')
    match mode:
        case 'behaviour':
            return build_behaviour_topology(broker)
        case 'activity':
            return build_activity_topology(broker)
        case 'presence':
            return build_presence_topology(broker)
        case _:
            raise ValueError(f'Unsupported analysis mode: {mode}')


def build_behaviour_topology(broker: MessageBroker) -> Topology:
    fps = 24
    window_size = fps * 2
    window_step = fps // 2

    classifier = CompositeBehaviouralClassifier(node_features=13)

    probability_threshold: float = 0.6  # default threshold
    if 'SINK_PROBABILITY_THRESHOLD' in os.environ:
        threshold = os.environ['SINK_PROBABILITY_THRESHOLD']
        if bool(re.match(r'^[01]\.\d*$', threshold)):
            probability_threshold: float = float(threshold)

    builder = StreamsBuilder(broker)

    builder.stream('video_source') \
        .named('pose-detection-app') \
        .process(PoseDetector()) \
        .to('pose_detection_results')

    builder.stream('video_source') \
        .named('object-detection-app') \
        .process(ObjectDetector()) \
        .to('object_detection_results')

    builder.stream('video_source') \
        .named('dimensions-setter-app') \
        .for_each(DimensionsSetter(classifier).process)

    builder.stream('video_source') \
        .join('pose_detection_results', lambda frame, poses: {'video_source': frame, 'pose_detection_results': poses}) \
        .named('activity-recognition-app') \
        .process(SubRegionExtractor()) \
        .window(size=window_size, step=window_step) \
        .process(MultiPersonActivityRecognitionAnalyzer()) \
        .to('activity_detection_results')

    builder.stream('object_detection_results') \
        .join('pose_detection_results', lambda objects, poses: {'object_detection_results': objects, 'pose_detection_results': poses}) \
        .named('human-object-interaction-app') \
        .process(HumanObjectInteractionAnalyzer()) \
        .window(size=window_size, step=window_step) \
        .to('hoi_results')

    builder.stream('pose_detection_results') \
        .window(size=window_size, step=window_step) \
        .join('activity_detection_results', lambda poses, activity: {'pose_detection_results': poses, 'activity_detection_results': activity}) \
        .join('hoi_results', lambda other, hoi: {**other, 'hoi_results': hoi}) \
        .named('graph-lstm-classifier-app') \
        .process(classifier) \
        .filter(lambda probability: probability > probability_threshold) \
        .to('output')

    return builder.build()


def build_activity_topology(broker: MessageBroker) -> Topology:
    fps = 24
    window_size = fps * 2
    window_step = fps // 2

    builder = StreamsBuilder(broker)

    builder.stream('video_source') \
        .named('pose-detection-app') \
        .process(PoseDetector()) \
        .to('pose_detection_results')

    builder.stream('video_source') \
        .join('pose_detection_results', lambda frame, poses: {'video_source': frame, 'pose_detection_results': poses}) \
        .named('activity-recognition-app') \
        .process(SubRegionExtractor()) \
        .window(size=window_size, step=window_step) \
        .process(MultiPersonActivityRecognitionAnalyzer()) \
        .through('activity_detection_results') \
        .process(SuspiciousActivityClassifier()) \
        .to('output')

    return builder.build()


def build_presence_topology(broker: MessageBroker) -> Topology:
    builder = StreamsBuilder(broker)

    builder.stream('video_source') \
        .named('simple-presence-classification-app') \
        .process(PoseDetector()) \
        .barrier(3, lambda results: len(results) > 0) \
        .to('output')

    return builder.build()


def main(argv: list[str]):
    video_url, analysis_mode, notification_service_url = argv
    broker = MessageBroker()
    topology = build_topology_for(analysis_mode, broker)
    topology.add_source('video_source', VideoSourceProducer(broker, video_url))
    topology.add_sink('output', NotificationSink(broker, notification_service_url))

    signal.signal(signal.SIGINT, lambda signum, frame: broker.interrupt())
    signal.signal(signal.SIGTERM, lambda signum, frame: broker.interrupt())

    streams_app = KafkaStreams(topology)
    streams_app.start()
    streams_app.wait()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        logging.error("Invalid command-line arguments. Required <video_url> <classification_type> <notification_webhook>")
        sys.exit(1)

    setup_logger()
    main(sys.argv[1:])
