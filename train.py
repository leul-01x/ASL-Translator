import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os

DATA_DIR = './data'
letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

dfs = []
letter_counts = []

# Load all CSVs and track per-letter usable samples
for idx, letter in enumerate(letters):
    path = os.path.join(DATA_DIR, f'class_{idx}.csv')
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping.")
        continue
    df = pd.read_csv(path)
    df = df.dropna()
    dfs.append(df)
    letter_counts.append((letter, len(df)))

# Show per-letter sample counts
print("Usable samples per letter:")
for letter, count in letter_counts:
    print(f"{letter}: {count}")

# Combine all letters
data = pd.concat(dfs, ignore_index=True)
data = data.dropna()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Total usable samples:", len(data))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN model
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Model accuracy on test data: {accuracy*100:.2f}%")