import os
from rtmlib import RTMO


class PoseDetector:

    def __init__(self):
        self.model = RTMO(os.environ["RTMO_MODEL_URL"])

    # TODO: only run the model if YOLO has detected people
    # otherwise, return zeros
