import cv2
import numpy as np
from analysis.double_frame_analyzer import DoubleFrameAnalyzer


class OpticalFlowAnalyzer(DoubleFrameAnalyzer):

    def analyze_with_previous_frame(self, frame1: cv2.typing.MatLike, frame2: cv2.typing.MatLike) -> list[int]:
        prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # Calculate optical flow using the Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute magnitude and angle of the optical flow vectors
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        # Flatten magnitude and angle arrays into feature vector
        mag_feature = mag.flatten()
        ang_feature = ang.flatten()
        # Concatenate magnitude and angle features
        return np.concatenate((mag_feature, ang_feature)).tolist()
