from enum import IntEnum


class AnalysisType(IntEnum):
    PersonDetection = 1
    HumanObjectInteraction = 2
    PoseEstimation = 3
    OpticalFlow = 4
    TemporalDifferenceWithHOG = 5
    ActivityDetection = 6
