import os
import json
import pandas as pd
import numpy as np
from datetime import datetime


class KeystrokeProcessor:
    def __init__(self, raw_dir="keystroke_data/raw", processed_dir="keystroke_data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def _extract_features(self, events, user):
        key_down_times = {}
        dwell_times = []
        flight_times = []
        last_down_time = None

        for event in events:
            key = event.get("name")
            event_type = event.get("event_type")
            t = event.get("time")

            if key is None or event_type is None or t is None:
                continue

            if key.startswith("Key."):
                continue

            if event_type == "down":
                key_down_times[key] = t
                if last_down_time is not None:
                    flight_times.append(t - last_down_time)
                last_down_time = t

            elif event_type == "up" and key in key_down_times:
                dwell_times.append(t - key_down_times[key])

        if len(dwell_times) < 3 or len(flight_times) < 3:
            return None

        return {
            "dwell_mean": np.mean(dwell_times),
            "dwell_std": np.std(dwell_times),
            "flight_mean": np.mean(flight_times),
            "flight_std": np.std(flight_times),
            "num_events": len(events),
            "user": user
        }

    def create_dataset(self):
        rows = []

        for filename in os.listdir(self.raw_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.raw_dir, filename)

            with open(filepath, "r") as f:
                events = json.load(f)

            user = filename.split("_")[0]
            features = self._extract_features(events, user)

            if features:
                rows.append(features)

        if not rows:
            print("No valid sessions found.")
            return None

        dataset = pd.DataFrame(rows)

        output_path = os.path.join(
            self.processed_dir,
            f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        dataset.to_csv(output_path, index=False)
        print("Saved dataset:", output_path)
        return output_path
