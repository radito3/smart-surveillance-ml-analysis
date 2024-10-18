import logging
import os
import sys

from context.topology_builder import TopologyBuilder
from messaging.message_broker import MessageBroker
from messaging.source.video_source_producer import VideoSourceProducer


def main(video_url: str, analysis_mode: str, notif_webhook_url: str):
    broker = TopologyBuilder.build_topology_for(analysis_mode, notif_webhook_url)

    source = VideoSourceProducer(broker, video_url)

    try:
        source.init()
    except Exception as e:
        logging.error(e)
        return

    broker.start()
    broker.wait()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        logging.error("Invalid command-line arguments. Required <video_url> <classification_type> <notification_webhook>")
        sys.exit(1)

    video_url_ = sys.argv[1]
    analysis_mode_ = sys.argv[2]
    notification_webhook = sys.argv[3]

    log_level_env = os.environ['LOG_LEVEL']
    mapping = logging.getLevelNamesMapping()
    if log_level_env is not None and len(log_level_env) != 0 and log_level_env in mapping:
        logging.root.setLevel(mapping[log_level_env])
    else:
        logging.root.setLevel(logging.DEBUG)

    main(video_url_, analysis_mode_, notification_webhook)
