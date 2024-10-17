import cv2
import numpy as np


class OpticalFlowAnalyzer:

    def __init__(self):
        self._pref_frame: cv2.typing.MatLike = None

    def analyze(self, frame: cv2.typing.MatLike) -> list[any]:
        if self._pref_frame is None:
            self._pref_frame = frame
            return []
        result = self.analyze_with_previous_frame(self._pref_frame, frame)
        self._pref_frame = frame
        return result

    # def analysis_type(self) -> AnalysisType:
    #     return AnalysisType.OpticalFlow

    # FIXME: this produces a huge list for each frame; consider removing / only using it in a special case
    def analyze_with_previous_frame(self, previous: cv2.typing.MatLike, current: cv2.typing.MatLike) -> list[any]:
        prev = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        # Calculate optical flow using the Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute magnitude and angle of the optical flow vectors
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        # Flatten magnitude and angle arrays into feature vector
        mag_feature = mag.flatten()
        ang_feature = ang.flatten()
        # Concatenate magnitude and angle features
        return np.concatenate((mag_feature, ang_feature)).tolist()
