import cv2
# from skimage.feature import hog as histogram_of_oriented_gradients
# from analysis.double_frame_analyzer import DoubleFrameAnalyzer
# from analysis.types import AnalysisType


class TemporalDifferenceAnalyzer:

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
    #     return AnalysisType.TemporalDifferenceWithHOG

    def analyze_with_previous_frame(self, previous: cv2.typing.MatLike, current: cv2.typing.MatLike) -> list[any]:
        # Convert frames to grayscale
        gray_frame1 = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        # This is mathematically and computationally simpler than skimage.metrics.structural_similarity
        # and is more appropriate for the current use case
        # https://ece.uwaterloo.ca/%7Ez70wang/publications/ssim.pdf
        diff = cv2.absdiff(gray_frame1, gray_frame2)
        return self.histogram_of_oriented_gradients(diff)

    def histogram_of_oriented_gradients(self, diff) -> any:
        # use a library
        pass
