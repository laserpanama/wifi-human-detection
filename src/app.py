from flask import Flask, render_template, jsonify, request
from collections import deque
import threading
import time

# Use relative imports if part of a package, 
# but for a simple script run directly, we might need to adjust PYTHONPATH.
# For now, let's keep them absolute but ensure they work.
try:
    from .simulator import WiFiSimulator
    from .processor import SignalProcessor
    from .detector import HumanDetector
except ImportError:
    from simulator import WiFiSimulator
    from processor import SignalProcessor
    from detector import HumanDetector

app = Flask(__name__)

# Global instances
simulator = WiFiSimulator()
processor = SignalProcessor(window_size=20)
detector = HumanDetector()

# Data storage for visualization - using deque for O(1) append/pop
history = deque(maxlen=100)

def background_worker():
    while True:
        reading = simulator.get_next_reading()
        processor.add_reading(reading)
        features = processor.get_features()
        is_present = detector.detect(features)
        
        data_point = {
            "timestamp": float(time.time()),
            "raw_rssi": float(reading),
            "filtered_rssi": float(features["mean"]),
            "variance": float(features["variance"]),
            "is_present": bool(is_present)
        }
        
        history.append(data_point)
        time.sleep(0.5)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/data")
def get_data():
    return jsonify(list(history))

@app.route("/api/toggle_human", methods=["POST"])
def toggle_human():
    present = request.json.get("present", False)
    simulator.set_human_presence(present)
    return jsonify({"status": "success", "human_present": present})

if __name__ == "__main__":
    # Start background worker
    threading.Thread(target=background_worker, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
