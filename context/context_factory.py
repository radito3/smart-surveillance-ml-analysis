import os
from datetime import timedelta
import torch
from analysis.activity.multi_person_activity_recon import MultiPersonActivityRecognitionAnalyzer
from classification.activity.suspicious_activity_classifier import SuspiciousActivityClassifier
from classification.behavior.graph_lstm import GraphBasedLSTMClassifier
from classification.people_presence.simple_presence_classifier import SimplePresenceClassifier
from analysis.object_detection.detector import ObjectDetector
from analysis.pose_detection.pose_detector import PoseDetector
from messaging.message_broker import MessageBroker
from messaging.sink.binary_result_consumer import BinaryResultConsumer
from messaging.sink.probability_result_consumer import ProbabilityResultConsumer
from util.device import get_device


class TopologyBuilder:

    @staticmethod
    def build_topology_for(mode: str, notification_webhook_url: str) -> MessageBroker:
        broker = MessageBroker()
        match mode:
            case "behaviour":
                object_detector = ObjectDetector(broker)
                pose_detector = PoseDetector(broker)
                activity_detector = MultiPersonActivityRecognitionAnalyzer(broker, 24, timedelta(seconds=2), 12)

                classifier = GraphBasedLSTMClassifier(broker, 5, 48, 12)
                pretrained_weights_path = os.environ['GRAPH_LSTM_WEIGHTS_PATH']
                if pretrained_weights_path is not None and len(pretrained_weights_path) != 0:
                    classifier.load_state_dict(torch.load(pretrained_weights_path, map_location=get_device()))
                # only on CUDA due to: https://github.com/pytorch/pytorch/issues/125254
                classifier.compile() if torch.cuda.is_available() else None

                sink = ProbabilityResultConsumer(broker, notification_webhook_url)

                topics = ['video_source', 'video_dimensions', 'object_detection_results', 'pose_detection_results',
                          'activity_detection_results', 'classification_results']
                for topic in topics:
                    broker.create_topic(topic)

                broker.add_subscriber_for('video_source', object_detector)
                broker.add_subscriber_for('video_source', pose_detector)
                broker.add_subscriber_for('video_source', activity_detector)
                broker.add_subscriber_for('object_detection_results', activity_detector)
                broker.add_subscriber_for('object_detection_results', classifier)
                broker.add_subscriber_for('pose_detection_results', classifier)
                broker.add_subscriber_for('activity_detection_results', classifier)
                broker.add_subscriber_for('classification_results', sink)
            case "activity":
                object_detector = ObjectDetector(broker)
                activity_detector = MultiPersonActivityRecognitionAnalyzer(broker, 24, timedelta(seconds=2), 12)

                classifier = SuspiciousActivityClassifier(broker)

                sink = BinaryResultConsumer(broker, notification_webhook_url)

                topics = ['video_source', 'video_dimensions', 'object_detection_results', 'activity_detection_results',
                          'classification_results']
                for topic in topics:
                    broker.create_topic(topic)

                broker.add_subscriber_for('video_source', object_detector)
                broker.add_subscriber_for('video_source', activity_detector)
                broker.add_subscriber_for('object_detection_results', activity_detector)
                broker.add_subscriber_for('activity_detection_results', classifier)
                broker.add_subscriber_for('classification_results', sink)
            case "presence":
                object_detector = ObjectDetector(broker)

                classifier = SimplePresenceClassifier(broker)

                sink = BinaryResultConsumer(broker, notification_webhook_url)

                topics = ['video_source', 'video_dimensions', 'object_detection_results', 'classification_results']
                for topic in topics:
                    broker.create_topic(topic)

                broker.add_subscriber_for('video_source', object_detector)
                broker.add_subscriber_for('object_detection_results', classifier)
                broker.add_subscriber_for('classification_results', sink)
        return broker
