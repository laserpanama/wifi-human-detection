class HumanDetector:
    def __init__(self, rssi_threshold=-55, variance_threshold=5.0):
        self.rssi_threshold = rssi_threshold
        self.variance_threshold = variance_threshold

    def detect(self, features):
        """
        Simple heuristic detection:
        - If RSSI drops below threshold OR
        - If variance exceeds threshold
        Then human is likely present.
        """
        mean_rssi = features.get("mean", 0)
        variance = features.get("variance", 0)

        # In a real scenario, these thresholds would be calibrated
        is_present = mean_rssi < self.rssi_threshold or variance > self.variance_threshold
        
        return is_present
