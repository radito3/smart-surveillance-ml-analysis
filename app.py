import logging
import os
import signal
import sys
from threading import Thread

from messaging.message_broker import MessageBroker
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
    video_url, analysis_mode, notification_service_url = argv
    topology_builder = TopologyBuilder()
    broker = MessageBroker()

    source = VideoSourceProducer(broker, video_url)
    streams = topology_builder.build_topology_for(analysis_mode, broker, notification_service_url)
    try:
        source.start()
    except Exception as e:
        logging.error(e)
        return

    signal.signal(signal.SIGINT, lambda signum, frame: broker.interrupt())
    signal.signal(signal.SIGTERM, lambda signum, frame: broker.interrupt())

    stream_threads = [Thread(name=stream.name, target=stream.run) for stream in streams]
    [thread.start() for thread in stream_threads]
    [thread.join() for thread in stream_threads]


if __name__ == '__main__':
    if len(sys.argv) != 4:
        logging.error("Invalid command-line arguments. Required <video_url> <classification_type> <notification_webhook>")
        sys.exit(1)

    setup_logger()
    main(sys.argv[1:])
