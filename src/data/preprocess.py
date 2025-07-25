import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load cleaned data

df = pd.read_csv("../../data/processed/eeg_clean.csv")
print(f'Loaded Cleaned dataset with shape:{df.shape}')

# Separate features and target

X = df.drop("eyeDetection",axis=1)
y = df["eyeDetection"]

# Train and Test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save to CSV

os.makedirs("data/processed", exist_ok=True)
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("data/processed/X_train.csv",index=False)
pd.DataFrame(X_test_scaled,columns=X.columns).to_csv("data/processed/X_test.csv",index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print(" preprocessing completed successfully and saved to data/processed/")