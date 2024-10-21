import logging
import os
import signal
import sys

from messaging.topology.topology_builder import TopologyBuilder
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


def main(argv: list[str]):
    video_url, analysis_mode, notification_webhook = argv

    broker = TopologyBuilder.build_topology_for(analysis_mode, notification_webhook)

    source = VideoSourceProducer(broker, video_url)
    try:
        source.init()
    except Exception as e:
        logging.error(e)
        return

    signal.signal(signal.SIGINT, lambda signum, frame: broker.interrupt())
    signal.signal(signal.SIGTERM, lambda signum, frame: broker.interrupt())

    broker.start_streams()
    broker.wait()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        logging.error("Invalid command-line arguments. Required <video_url> <classification_type> <notification_webhook>")
        sys.exit(1)

    setup_logger()
    main(sys.argv[1:])
