import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

DATA_DIR = './data'
letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

dfs = []
for idx, letter in enumerate(letters):
    path = os.path.join(DATA_DIR, f'class_{idx}.csv')
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    df = df.dropna()
    dfs.append(df)

if not dfs:
    print("No data found. Run your collect.py first.")
    exit()

data = pd.concat(dfs, ignore_index=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X, y)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label_map = {i: letters[i] for i in range(len(letters))}

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = []

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:
            landmarks_np = np.array(landmarks).reshape(1, -1)
            prediction = clf.predict(landmarks_np)[0]
            letter = label_map[int(prediction)]
            cv2.putText(frame, f'Letter: {letter}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow('ASL Live', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
