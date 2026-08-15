import time


class Timer:
    """Mede duração em milissegundos."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self._start: float = 0
        self.elapsed_ms: float = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        return False
