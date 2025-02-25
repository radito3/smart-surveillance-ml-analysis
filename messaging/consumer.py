import logging


class Consumer:

    def get_name(self) -> str:
        return 'base-consumer'

    def init(self):
        pass

    def process_message(self, message: any):
        logging.warning('Process called on a base consumer. Skipping message...')

    def cleanup(self):
        pass
