import unittest
from src.processor import SignalProcessor
from src.detector import HumanDetector

class TestWifiDetection(unittest.TestCase):
    def test_processor_average(self):
        processor = SignalProcessor(window_size=5)
        for i in range(1, 6):
            processor.add_reading(i)
        self.assertEqual(processor.get_filtered_signal(), 3.0)

    def test_detector_presence(self):
        detector = HumanDetector(rssi_threshold=-55, variance_threshold=5.0)
        
        # Test absent (high RSSI, low variance)
        features_absent = {"mean": -50, "variance": 1.0}
        self.assertFalse(detector.detect(features_absent))
        
        # Test present due to RSSI drop
        features_rssi_drop = {"mean": -60, "variance": 1.0}
        self.assertTrue(detector.detect(features_rssi_drop))
        
        # Test present due to variance increase
        features_high_variance = {"mean": -50, "variance": 10.0}
        self.assertTrue(detector.detect(features_high_variance))

if __name__ == "__main__":
    unittest.main()
