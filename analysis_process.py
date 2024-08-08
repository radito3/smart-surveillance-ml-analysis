import multiprocessing as mp
import multiprocessing.connection as cn
import cv2

from analysis.analyzer import BaseAnalyzer


def analyzer_wrapper(analyzer: BaseAnalyzer, frame_src: cn.Connection, sink: cn.Connection, sink_lock: mp.Lock):
    while True:
        # should there be a timeout when reading?
        frame: cv2.typing.MatLike = frame_src.recv()
        if frame is None:
            return

        vector: list[any] = analyzer.analyze(frame)

        sink_lock.acquire()
        try:
            sink.send({"type": analyzer.analysis_type(), "data": vector})
        finally:
            sink_lock.release()
