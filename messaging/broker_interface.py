from abc import ABC, abstractmethod


class Broker(ABC):

    @abstractmethod
    def read_from(self, topic: str) -> any:
        pass

    @abstractmethod
    def write_to(self, topic: str, message: any):
        pass
