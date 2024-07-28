import cv2
from skimage.feature import hog as histogram_of_oriented_gradients
from analysis.double_frame_analyzer import DoubleFrameAnalyzer


class TemporalDifferenceAnalyzer(DoubleFrameAnalyzer):

    def analyze(self, frame1: cv2.typing.MatLike, *args, **kwargs) -> list[int]:
        # Convert frames to grayscale
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(*args[0], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_frame1, gray_frame2)
        return histogram_of_oriented_gradients(diff,
                                               # (for colored images)
                                               # indicates which axis of the array corresponds to channels
                                               channel_axis=-1)

    def get_num_frames(self) -> int:
        return 2
