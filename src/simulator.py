import numpy as np
import time

class WiFiSimulator:
    def __init__(self, base_rssi=-50, noise_level=1.0):
        self.base_rssi = base_rssi
        self.noise_level = noise_level
        self.is_human_present = False

    def set_human_presence(self, present: bool):
        self.is_human_present = present

    def get_next_reading(self):
        """
        Generates the next RSSI reading.
        If a human is present, the signal drops and variance increases.
        """
        rssi = self.base_rssi
        noise = np.random.normal(0, self.noise_level)
        
        if self.is_human_present:
            # Simulate signal absorption (drop in RSSI) and multipath (increased noise)
            rssi -= 10  # 10dB drop
            noise = np.random.normal(0, self.noise_level * 3)
            
        return rssi + noise

if __name__ == "__main__":
    simulator = WiFiSimulator()
    for i in range(10):
        if i == 5:
            simulator.set_human_presence(True)
        print(f"Reading {i}: {simulator.get_next_reading():.2f}")
