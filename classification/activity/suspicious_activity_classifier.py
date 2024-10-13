import logging
import os
from analysis.types import AnalysisType
from classification.classifier import Classifier


class SuspiciousActivityClassifier(Classifier):

    def __init__(self):
        env_whitelist = os.environ['ACTIVITY_WHITELIST']
        self.whitelist_activities_indices = []
        if env_whitelist is not None and len(env_whitelist) != 0:
            for env_line in env_whitelist:
                parsed = self.__parse_env_line(env_line)
                if isinstance(parsed, int):
                    self.whitelist_activities_indices.append(parsed)
                else:
                    self.whitelist_activities_indices.extend(parsed)
        else:
            self.whitelist_activities_indices = [2, 8]  # temp

    def classify_as_suspicious(self, dtype: AnalysisType, vector: list[any]) -> float:
        if dtype != AnalysisType.ActivityDetection or len(vector) == 0:
            return 0

        for activity_idx in vector:
            if activity_idx not in self.whitelist_activities_indices:
                return 1
        return 0

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
