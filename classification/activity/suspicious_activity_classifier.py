import logging
import os

from messaging.processor import MessageProcessor


class SuspiciousActivityClassifier(MessageProcessor):

    def __init__(self):
        super().__init__()
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

    def process(self, people_activities: list[int]):
        for activity_idx in people_activities:
            if activity_idx not in self.whitelist_activities_indices:
                self.next(True)

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
