import multiprocessing as mp


class ReceiverLock:
    def __init__(self, lock: mp.Lock):
        self.lock = lock

    def __enter__(self) -> None:
        self.lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.lock.release()
