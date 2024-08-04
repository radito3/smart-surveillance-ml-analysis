import cv2
from skimage.feature import hog as histogram_of_oriented_gradients
from analysis.double_frame_analyzer import DoubleFrameAnalyzer
from analysis.types import AnalysisType


class TemporalDifferenceAnalyzer(DoubleFrameAnalyzer):

    def analysis_type(self) -> AnalysisType:
        return AnalysisType.TemporalDifferenceWithHOG

    def analyze_with_previous_frame(self, frame1: cv2.typing.MatLike, frame2: cv2.typing.MatLike) -> list[int]:
        # Convert frames to grayscale
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # This is mathematically and computationally simpler than skimage.metrics.structural_similarity
        # and is more appropriate for the current use case
        # https://ece.uwaterloo.ca/%7Ez70wang/publications/ssim.pdf
        diff = cv2.absdiff(gray_frame1, gray_frame2)
        return histogram_of_oriented_gradients(diff)
