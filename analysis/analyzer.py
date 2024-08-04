from abc import ABC, abstractmethod


class BaseAnalyzer(ABC):

    @abstractmethod
    def get_num_frames(self) -> int:
        pass

    @abstractmethod
    def analyze(self, payload: any, *args, **kwargs) -> list[int]:
        raise AssertionError("abstract class method called")

    def __call__(self, payload: any, *args, **kwargs) -> list[int]:
        return self.analyze(payload, *args, **kwargs)
