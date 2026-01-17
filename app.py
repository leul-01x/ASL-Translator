import cv2                   
import mediapipe as mp       
import pandas as pd          
from sklearn.neighbors import KNeighborsClassifier
import numpy as np           


df_a = pd.read_csv('./data/A.csv')  
df_b = pd.read_csv('./data/B.csv')  
df_c = pd.read_csv('./data/C.csv')  
df_d = pd.read_csv('./data/D.csv')  
df_e = pd.read_csv('./data/E.csv')  

data = pd.concat([df_a, df_b, df_c, df_d, df_e], ignore_index=True)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X, y)  

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils  

cap = cv2.VideoCapture(0)  

label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

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

        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks_np = np.array(landmarks).reshape(1, -1)

        prediction = clf.predict(landmarks_np)[0]
        letter = label_map[int(prediction)]

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f'Letter: {letter}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow('ASL Live', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
