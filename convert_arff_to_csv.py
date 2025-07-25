from scipy.io import arff
import pandas as pd


data, meta = arff.loadarff("data/raw/EEG Eye State.arff")
df = pd.DataFrame(data)
csv_path = f'data/raw/eeg_eye_state.csv'
df.to_csv(csv_path, index= False)

print(" EEG dataset converted and saved to {csv_path}")