import logging
import os

from messaging.broker_interface import Broker
from messaging.consumer import Consumer
from messaging.producer import Producer


class SuspiciousActivityClassifier(Producer, Consumer):

    def __init__(self, broker: Broker):
        Producer.__init__(self, broker)
        Consumer.__init__(self, broker, 'activity_detection_results')

        self.whitelist_activities_indices = []
        if 'ACTIVITY_WHITELIST' in os.environ:
            env_whitelist = os.environ['ACTIVITY_WHITELIST']
            for env_line in env_whitelist.split(','):
                parsed = self.__parse_env_line(env_line)
                if isinstance(parsed, int):
                    self.whitelist_activities_indices.append(parsed)
                else:
                    self.whitelist_activities_indices.extend(parsed)
        else:
            self.whitelist_activities_indices = [2, 8]  # temp

    def get_name(self) -> str:
        return 'suspicious-activity-classifier-app'

    def consume_message(self, people_activities: list[int]):
        for activity_idx in people_activities:
            if activity_idx not in self.whitelist_activities_indices:
                self.produce_value('classification_results', True)

    def cleanup(self):
        self.produce_value('classification_results', None)

    @staticmethod
    def __parse_env_line(value: str) -> int | list[int]:
        if '-' in value:
            bounds = value.split('-')
            if len(bounds) != 2:
                logging.error(f"Activity whitelist format error: {value}. Skipping value")
                return []
            return [i for i in range(int(bounds[0]), int(bounds[1]) + 1)]
        else:
            return int(value)
