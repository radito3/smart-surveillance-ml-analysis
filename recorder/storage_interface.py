from abc import ABC, abstractmethod


class Storage(ABC):

    @abstractmethod
    def read(self) -> bytearray:
        pass

    @abstractmethod
    def write(self, buff: bytearray, offset: int, length: int) -> None:
        pass
