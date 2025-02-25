import logging
import os

from messaging.broker_interface import Broker
from messaging.consumer import Consumer
from messaging.producer import Producer


class SuspiciousActivityClassifier(Producer, Consumer):

    def __init__(self, broker: Broker):
        Producer.__init__(self, broker)

        self.whitelist_activities_indices: set[int] = set()
        if 'ACTIVITY_WHITELIST' in os.environ:
            env_whitelist = os.environ['ACTIVITY_WHITELIST']
            for env_line in env_whitelist.split(','):
                parsed = self.__parse_env_line(env_line)
                if isinstance(parsed, int):
                    self.whitelist_activities_indices.add(parsed)
                else:
                    self.whitelist_activities_indices.update(parsed)
        else:
            self.whitelist_activities_indices = {2, 8}  # sample activity indices

    def get_name(self) -> str:
        return 'suspicious-activity-classifier-app'

    def process_message(self, people_activities: list[int]):
        for activity_idx in people_activities:
            if activity_idx not in self.whitelist_activities_indices:
                self.publish('classification_results', True)

    def cleanup(self):
        self.publish('classification_results', None)

    @staticmethod
    def __parse_env_line(value: str) -> int | set[int]:
        if '-' in value:
            bounds = value.split('-')
            if len(bounds) != 2:
                logging.error(f"Activity whitelist format error: {value}. Skipping value")
                return set()
            return {i for i in range(int(bounds[0]), int(bounds[1]) + 1)}
        else:
            return int(value)
