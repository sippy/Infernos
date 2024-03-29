from threading import Lock
from time import monotonic
from math import pi as Pi

class InfernTorcherDeadlock(Exception):
    pass

class rc_filter():
    alpha: float
    last_y: float

    def __init__(self, x = 10, init_y = 0.0):
        self.alpha = 1 / (1 + 2 * Pi * x)
        self.last_y = init_y
    
    def __call__(self, x):
        self.last_y = self.alpha * x + (1 - self.alpha) * self.last_y
        return self.last_y

class InfernTorcher():
    _torch_lock: Lock = None
    _last_lock: float
    _last_unlock: float
    _free_time: rc_filter
    _busy_time: rc_filter
    _nlocks: int = 0

    def __init__(self):
        self._torch_lock = Lock()
        self._last_unlock = self._last_lock = monotonic()
        self._free_time = rc_filter()
        self._busy_time = rc_filter()

    def lock(self, timeout: int = 10):
        acquired = self._torch_lock.acquire(timeout = timeout)
        if not acquired:
            raise InfernTorcherDeadlock(f"Could not acquire lock within {timeout} seconds")
        now = monotonic()
        free_time = now - self._last_unlock
        self._free_time(free_time)
        self._last_lock = now

    def unlock(self):
        now = monotonic()
        busy_time = now - self._last_lock
        bt = self._busy_time(busy_time)
        ft = self._free_time.last_y
        self._last_unlock = now
        self._nlocks += 1
        nlocks = self._nlocks
        self._torch_lock.release()
        if (nlocks % 100) == 0:
            print(f"Torch load: {bt / (bt + ft)}")

    def acquire(self):
        return self.lock()
    
    def release(self):
        return self.unlock()

    def __enter__(self):
        self.lock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unlock()