import pandas as pd
import glob
from keystroke_identifier import KeystrokeIdentifier

csv_files = glob.glob("keystroke_data/processed/*.csv")
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

if 'user' not in df.columns and 'user_id' in df.columns:
    df.rename(columns={'user_id': 'user'}, inplace=True)

df.to_csv("keystroke_data/processed/dataset_combined.csv", index=False)
print("Combined dataset saved.")

ki = KeystrokeIdentifier()
ki.train_model("keystroke_data/processed/dataset_combined.csv")
