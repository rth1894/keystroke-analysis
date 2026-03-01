import json
import time
import os
from datetime import datetime
from pynput import keyboard

class KeystrokeCollector:
    def __init__(self, output_dir="keystroke_data"):
        self.output_dir = output_dir
        self.current_session = []
        self.last_keystroke_time = None
        self.session_timeout = 2.0
        self.user_id = None
        self.session_id = None
        self.listener = None

        os.makedirs(f"{output_dir}/raw", exist_ok=True)
        os.makedirs(f"{output_dir}/processed", exist_ok=True)

    def start_collection(self, user_id="unknown"):
        self.user_id = user_id
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"Starting keystroke collection for user: {user_id}")
        print("Press ESC to stop collection")

        def on_press(key):
            self._handle_event(key, "down")

        def on_release(key):
            self._handle_event(key, "up")
            if key == keyboard.Key.esc:
                return False  # stop listener

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            self.listener = listener
            listener.join()

        if self.current_session:
            self._save_current_session()

        print("Collection stopped.")

    def _handle_event(self, key, event_type):
        current_time = time.time()

        if self.last_keystroke_time is not None and \
           (current_time - self.last_keystroke_time) > self.session_timeout:
            self._save_current_session()
            self.current_session = []

        self.last_keystroke_time = current_time

        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key)

        self.current_session.append({
            "event_type": event_type,
            "name": key_name,
            "time": current_time
        })

    def _save_current_session(self):
        if len(self.current_session) < 10:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/raw/{self.user_id}_{self.session_id}_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(self.current_session, f, indent=2)

        print(f"Saved session with {len(self.current_session)} keystrokes")
