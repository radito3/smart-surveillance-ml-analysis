import logging
from datetime import timedelta
from analysis.activity.multi_person_activity_recon import MultiPersonActivityRecognitionAnalyzer
from analysis.human_object_interaction.interaction import HumanObjectInteractionAnalyzer
from classification.activity.suspicious_activity_classifier import SuspiciousActivityClassifier
from classification.behavior.graph_lstm import CompositeBehaviouralClassifier, GraphBasedLSTMClassifier
from classification.people_presence.simple_presence_classifier import SimplePresenceClassifier
from analysis.object_detection.object_detector import ObjectDetector
from analysis.pose_detection.pose_detector import PoseDetector
from messaging.message_broker import MessageBroker
from messaging.sink.binary_result_sink import BinaryResultSink
from messaging.sink.probability_result_sink import ProbabilityResultSink
from messaging.sink.training_sink import TrainingSink


class TopologyBuilder:

    @classmethod
    def build_topology_for(cls, mode: str, notification_webhook_url: str) -> MessageBroker:
        broker = MessageBroker()
        logging.debug(f"Creating streams topology for analysis mode: {mode}")
        match mode:
            case "behaviour":
                object_detector = ObjectDetector(broker)
                pose_detector = PoseDetector(broker)
                hoi_detector = HumanObjectInteractionAnalyzer(broker)
                activity_detector = MultiPersonActivityRecognitionAnalyzer(broker, 24, timedelta(seconds=2), 12)

                classifier = CompositeBehaviouralClassifier(broker, 12, 48, 12)

                sink = ProbabilityResultSink(broker, notification_webhook_url)

                topics = ['video_source', 'video_dimensions', 'object_detection_results', 'pose_detection_results',
                          'activity_detection_results', 'hoi_results', 'classification_results']
                for topic in topics:
                    logging.debug(f"Creating topic {topic}")
                    broker.create_topic(topic)

                broker.add_subscriber_for('video_source', object_detector)
                broker.add_subscriber_for('video_source', pose_detector)
                broker.add_subscriber_for('video_source', activity_detector)
                broker.add_subscriber_for('pose_detection_results', activity_detector)
                broker.add_subscriber_for('object_detection_results', hoi_detector)
                broker.add_subscriber_for('pose_detection_results', hoi_detector)
                broker.add_subscriber_for('pose_detection_results', classifier)
                broker.add_subscriber_for('hoi_results', classifier)
                broker.add_subscriber_for('activity_detection_results', classifier)
                broker.add_subscriber_for('classification_results', sink)
            case "activity":
                pose_detector = PoseDetector(broker)
                activity_detector = MultiPersonActivityRecognitionAnalyzer(broker, 24, timedelta(seconds=2), 12)

                classifier = SuspiciousActivityClassifier(broker)

                sink = BinaryResultSink(broker, notification_webhook_url)

                topics = ['video_source', 'video_dimensions', 'pose_detection_results', 'activity_detection_results',
                          'classification_results']
                for topic in topics:
                    logging.debug(f"Creating topic {topic}")
                    broker.create_topic(topic)

                broker.add_subscriber_for('video_source', pose_detector)
                broker.add_subscriber_for('video_source', activity_detector)
                broker.add_subscriber_for('pose_detection_results', activity_detector)
                broker.add_subscriber_for('activity_detection_results', classifier)
                broker.add_subscriber_for('classification_results', sink)
            case "presence":
                pose_detector = PoseDetector(broker)

                classifier = SimplePresenceClassifier(broker)

                sink = BinaryResultSink(broker, notification_webhook_url)

                topics = ['video_source', 'video_dimensions', 'pose_detection_results', 'classification_results']
                for topic in topics:
                    logging.debug(f"Creating topic {topic}")
                    broker.create_topic(topic)

                broker.add_subscriber_for('video_source', pose_detector)
                broker.add_subscriber_for('pose_detection_results', classifier)
                broker.add_subscriber_for('classification_results', sink)
            case _:
                raise ValueError(f"Unsupported analysis mode: {mode}")
        return broker

    @classmethod
    def build_training_topology(cls, fps: int, model: GraphBasedLSTMClassifier) -> tuple[MessageBroker, TrainingSink]:
        window_size: int = 2 * fps
        window_step: int = fps // 2

        broker = MessageBroker()
        object_detector = ObjectDetector(broker)
        pose_detector = PoseDetector(broker)
        hoi_detector = HumanObjectInteractionAnalyzer(broker)
        activity_detector = MultiPersonActivityRecognitionAnalyzer(broker, fps, timedelta(seconds=2), window_step)

        classifier = CompositeBehaviouralClassifier(broker, 12, window_size, window_step)
        classifier.inject_model(model)

        sink = TrainingSink(broker)

        topics = ['video_source', 'video_dimensions', 'object_detection_results', 'pose_detection_results',
                  'activity_detection_results', 'hoi_results', 'classification_results']
        for topic in topics:
            broker.create_topic(topic)

        broker.add_subscriber_for('video_source', object_detector)
        broker.add_subscriber_for('video_source', pose_detector)
        broker.add_subscriber_for('video_source', activity_detector)
        broker.add_subscriber_for('pose_detection_results', activity_detector)
        broker.add_subscriber_for('object_detection_results', hoi_detector)
        broker.add_subscriber_for('pose_detection_results', hoi_detector)
        broker.add_subscriber_for('pose_detection_results', classifier)
        broker.add_subscriber_for('hoi_results', classifier)
        broker.add_subscriber_for('activity_detection_results', classifier)
        broker.add_subscriber_for('classification_results', sink)
        return broker, sink
