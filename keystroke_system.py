import glob
import sys
from keystroke_processor import KeystrokeProcessor
from keystroke_identifier import KeystrokeIdentifier, RealTimeIdentifier
from keystroke_collector import KeystrokeCollector


def main():
    print("Running file:", __file__)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python keystroke_system.py process")
        print("  python keystroke_system.py train")
        print("  python keystroke_system.py identify")
        return

    command = sys.argv[1]
    print("Command received:", command)

    if command == "process":
        processor = KeystrokeProcessor()
        processor.create_dataset()

    elif command == "train":
        files = glob.glob("keystroke_data/processed/dataset_*.csv")
        if not files:
            print("No dataset found. Run process first.")
            return

    elif command == "identify":
        app = RealTimeIdentifier()
        app.start()

    elif command == "collect":
        user_id = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        collector = KeystrokeCollector()
        collector.start_collection(user_id=user_id)

    else:
        print("Unknown command.")


if __name__ == "__main__":
    main()
