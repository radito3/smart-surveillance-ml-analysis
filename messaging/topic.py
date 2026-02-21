from collections import deque
from threading import Event, Lock, Condition


class Topic:
    class Record:
        def __init__(self, payload, subscriber_count):
            self.payload = payload
            self.ref_count = subscriber_count
            self.lock = Lock()

        def decrement(self):
            with self.lock:
                self.ref_count -= 1
                return self.ref_count == 0

    def __init__(self, name, queue_size=100):
        self.name = name
        self.subscribers_num = 0
        self.records = deque()
        self.max_num_records = queue_size
        self.condition = Condition()
        self.shutdown = Event()

    def subscribe(self):
        self.subscribers_num += 1

    def unsubscribe(self):
        self.subscribers_num -= 1

    def publish(self, message: any):
        if self.shutdown.is_set():
            return

        with self.condition:
            # Block until there's space available
            while len(self.records) >= self.max_num_records:
                # do not use wait_for(predicate) because the shutdown flag may be set even when the queue is full
                self.condition.wait()
                if self.shutdown.is_set():
                    return

            self.records.append(self.Record(message, self.subscribers_num))
            self.condition.notify_all()  # Wake up waiting pollers

    def consume(self) -> any:
        if self.shutdown.is_set():
            return None

        with self.condition:
            # Wait until there's a record available
            while not self.records:
                self.condition.wait()
                if self.shutdown.is_set():
                    return None

            record = self.records[0]
            data = record.payload

            if record.decrement():  # Last subscriber to read it
                self.records.popleft()
                # Notify publishers that space is now available
                self.condition.notify_all()

            return data

    def stop_processing_messages(self):
        self.shutdown.set()
        with self.condition:
            self.condition.notify_all()

    def __repr__(self):
        return f"Topic {self.name}"
