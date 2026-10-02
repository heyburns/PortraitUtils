"""Small per-instance state, never a global image/model tensor cache."""
from collections import OrderedDict
import threading


class TransactionalCursor:
    """Caller holds lock across select/decode/commit; failure leaves state unchanged."""
    def __init__(self, capacity=32):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Cursor capacity must be a positive integer.")
        self.capacity = capacity
        self.positions = OrderedDict()
        self.lock = threading.RLock()

    @staticmethod
    def next_item(items, previous=None, *, repeat=False, reverse=False):
        if not items:
            raise ValueError("No files are available for selection; cursor has not advanced.")
        if previous in items:
            if repeat:
                return previous
            return items[(items.index(previous) + (-1 if reverse else 1)) % len(items)]
        return items[-1] if reverse else items[0]

    def select(self, key, items, *, repeat=False, reverse=False):
        with self.lock:
            return self.next_item(items, self.positions.get(key), repeat=repeat, reverse=reverse)

    def commit(self, key, value):
        with self.lock:
            self.positions[key] = value
            self.positions.move_to_end(key)
            while len(self.positions) > self.capacity:
                self.positions.popitem(last=False)
