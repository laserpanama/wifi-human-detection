import numpy as np
from collections import deque

class SignalProcessor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add_reading(self, reading):
        self.buffer.append(reading)

    def get_filtered_signal(self):
        if not self.buffer:
            return 0
        return np.mean(self.buffer)

    def get_variance(self):
        if len(self.buffer) < 2:
            return 0
        return np.var(self.buffer)

    def get_features(self):
        return {
            "mean": self.get_filtered_signal(),
            "variance": self.get_variance()
        }
