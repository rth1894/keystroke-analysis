import pandas as pd
import numpy as np
import pickle
import time
import tkinter as tk
from threading import Thread
from pynput import keyboard
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class KeystrokeIdentifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None

    def train_model(self, dataset_path):
        data = pd.read_csv(dataset_path)

        if data["user"].nunique() < 2:
            print("Need at least 2 users to train classifier.")
            return

        X = data.drop(columns=["user"])
        y = data["user"]

        self.feature_names = X.columns.tolist()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)

        with open("keystroke_model.pkl", "wb") as f:
            pickle.dump({ "model": self.model, "scaler": self.scaler, "feature_names": self.feature_names }, f)

        print("Model trained and saved.")

    def load_model(self):
        with open("keystroke_model.pkl", "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_names = data["feature_names"]

    def identify_user(self, events):
        features = self._extract_features(events)
        df = pd.DataFrame([features])

        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_names]
        scaled = self.scaler.transform(df)

        user = self.model.predict(scaled)[0]
        probs = self.model.predict_proba(scaled)[0]

        confidence = { self.model.classes_[i]: float(probs[i]) for i in range(len(probs)) }

        return user, confidence

    def _extract_features(self, events):
        key_down_times = {}
        dwell = []
        flight = []
        last_down = None

        for e in events:
            key = e["name"]
            t = e["time"]

            if key.startswith("Key."):
                continue

            if e["event_type"] == "down":
                key_down_times[key] = t
                if last_down is not None:
                    flight.append(t - last_down)
                last_down = t

            elif e["event_type"] == "up" and key in key_down_times:
                dwell.append(t - key_down_times[key])

        return {
            "dwell_mean": np.mean(dwell) if dwell else 0,
            "dwell_std": np.std(dwell) if dwell else 0,
            "flight_mean": np.mean(flight) if flight else 0,
            "flight_std": np.std(flight) if flight else 0,
            "num_events": len(events)
        }


class RealTimeIdentifier:
    def __init__(self):
        self.identifier = KeystrokeIdentifier()
        self.identifier.load_model()

        self.window = []
        self.window_size = 50
        self.min_keys = 20

        self.root = tk.Tk()
        self.root.title("Keystroke Identifier")
        self.root.geometry("400x400")

        self.label = tk.Label(self.root, text="User: Unknown", font=("Arial", 16))
        self.label.pack(pady=20)

        Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()

    def _on_press(self, key):
        self._add("down", key)

    def _on_release(self, key):
        self._add("up", key)

    def _add(self, event_type, key):
        try:
            name = key.char
        except:
            name = str(key)

        self.window.append({ "event_type": event_type, "name": name, "time": time.time() })

        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]

        if len(self.window) >= self.min_keys:
            try:
                user, conf = self.identifier.identify_user(self.window)
                self.label.config(text=f"User: {user}")
                print("Prediction:", user, conf)
            except Exception as e:
                print("Prediction error:", e)

    def start(self):
        self.root.mainloop()
