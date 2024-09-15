import queue
import threading
import asyncio
from queue import Queue

from .analyzer import BaseAnalyzer
from .types import AnalysisType


class CacheAsideAnalyzer(BaseAnalyzer):

    def __init__(self, cache_queue: Queue, dtype: AnalysisType, delegate: BaseAnalyzer = None):
        self._delegate: BaseAnalyzer = delegate
        self._dtype: AnalysisType = dtype
        self._cache_queue: Queue = cache_queue
        self._exited: threading.Event = threading.Event()

    def analysis_type(self) -> AnalysisType:
        return self._dtype

    def analyze(self, payload: any, *args, **kwargs) -> list[any]:
        if self._exited.is_set():
            return []

        if self._delegate is not None:
            result = self._delegate.analyze(payload, *args, **kwargs)
            # print("caching value")
            try:
                self._cache_queue.put(result, timeout=3)
            except queue.Full:
                # self._exited.set()
                return []
            # print("cached")
            return result

        try:
            return self._cache_queue.get(timeout=3)
        except queue.Empty:
            return []

        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # result = loop.run_until_complete(self.waiter())
        # loop.close()
        # return result

    def stop(self) -> None:
        self._exited.set()

    async def waiter(self) -> list[any]:
        tasks = [asyncio.create_task(self.queue_not_empty()), asyncio.create_task(self.exited())]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        return done.result()

    async def queue_not_empty(self):
        self._cache_queue.not_empty.wait()
        return self._cache_queue.get_nowait()

    async def exited(self):
        self._exited.wait()
        return []
