from datetime import timedelta

import multiprocessing as mp
import multiprocessing.connection as cn
import cv2

from activity.activity_recon import ActivityRecognitionAnalyzer
from analysis.analyzer import BaseAnalyzer
from human_object_interaction.interaction import HumanObjectInteractionAnalyzer
from object_detection.detector import ObjectDetector
from optical_flow.optical_flow import OpticalFlowAnalyzer
from pose_detection.pose_detector import PoseDetector
from temporal_difference.temporal_difference import TemporalDifferenceAnalyzer


def analyzer_wrapper(analyzer: BaseAnalyzer, frame_src: cn.Connection, sink: cn.Connection, stop_signal: cn.Connection):
    while True:
        ready: list[cn.Connection] = cn.wait([frame_src, stop_signal])
        for conn in ready:
            if conn.fileno() == stop_signal.fileno():
                return

            frame: cv2.typing.MatLike = conn.recv()
            vector: list[int] = analyzer.analyze(frame)
            sink.send({"type": analyzer.analysis_type(), "data": vector})


def analysis(frames_src: cn.Connection, fps: int, window_size: timedelta, sink: cn.Connection,
             stop_signal: cn.Connection) -> None:
    # TODO: decide which analysis components to use
    # order is important, as that will be the vector format the LSTM model will be trained with
    analyzers: list[BaseAnalyzer] = [HumanObjectInteractionAnalyzer(ObjectDetector()), PoseDetector(),
                                     OpticalFlowAnalyzer(), TemporalDifferenceAnalyzer(),
                                     ActivityRecognitionAnalyzer(fps, window_size)]

    pipes: list[tuple[cn.Connection, cn.Connection]] = [mp.Pipe(duplex=False) for _ in analyzers]

    processes: list[mp.Process] = [mp.Process(target=analyzer_wrapper, args=(analyzer, src, sink, stop_signal,))
                                   for analyzer, (src, _) in zip(analyzers, pipes)]
    [process.start() for process in processes]

    while True:
        ready: list[cn.Connection] = cn.wait([frames_src, stop_signal])
        for conn in ready:
            if conn.fileno() == stop_signal.fileno():
                [process.join() for process in processes]
                return

        frame: cv2.typing.MatLike = frames_src.recv()
        [dest.send(frame) for _, dest in pipes]
