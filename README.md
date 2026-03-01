# keystroke-analysis

This project collects, processes, and analyzes keystroke data to identify users based on typing patterns. It includes:

- **Keystroke collection** (`keystroke_collector.py`)
- **Data preprocessing & dataset creation** (`keystroke_processor.py`)
- **Feature analysis and visualization** (`keystroke_analyzer.py`)
- **Real-time user identification** (`keystroke_identifier.py`, `keystroke_system.py`)

---

## Setup

1. **Install dependencies**:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn pynput
```
## Usage

1. **Collect keystrokes**

Run the collector for a user:
`python keystroke_system.py collect <user_id>`

Press ESC to stop the session.
Raw JSON files are stored in keystroke_data/raw/.

2. **Process raw keystrokes into features**
`python keystroke_system.py process`

Generates CSV files in keystroke_data/processed/.

3. **Train the model**
Combine processed CSVs and train:
`python keystroke_system.py train`

Saves keystroke_model.pkl in keystroke_data/models/.

4. **Identify user in real-time**
`python keystroke_system.py identify`

Opens a GUI that monitors keystrokes and predicts the current user.
