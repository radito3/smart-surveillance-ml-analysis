from queue import Queue, Empty, Full
from threading import Event

class Topic:

    def __init__(self, name, queue_size=100):
        self.subscribers: dict[str, Queue] = {}
        self.name = name
        self.queue_size = queue_size
        self.shutdown = Event()

    def subscribe(self, subscriber_id: str):
        if subscriber_id not in self.subscribers:
            self.subscribers[subscriber_id] = Queue(maxsize=self.queue_size)

    def unsubscribe(self, subscriber_id: str):
        if subscriber_id in self.subscribers:
            del self.subscribers[subscriber_id]

        if len(self.subscribers) == 0:
            self.shutdown.set()  # no point wasting resources when there are no downstream processors

    def publish(self, message: any) -> bool:
        if self.shutdown.is_set():
            return False

        for sub, queue in self.subscribers.items():
            if not self.__publish_to_queue(queue, message):
                return False

        return not self.shutdown.is_set()

    def __publish_to_queue(self, queue: Queue, message: any) -> bool:
        while not self.shutdown.is_set():
            try:
                queue.put(message, timeout=1)
                return True
            except Full:
                continue
            except KeyboardInterrupt:
                break
        return False

    def consume(self, subscriber_id: str) -> any:
        if subscriber_id in self.subscribers:
            while not self.shutdown.is_set():
                try:
                    return self.subscribers[subscriber_id].get(timeout=1)
                except Empty:
                    continue
                except KeyboardInterrupt:
                    break
        return None

    def stop_processing_messages(self):
        self.shutdown.set()

    def __repr__(self):
        return f'Topic {self.name}'
