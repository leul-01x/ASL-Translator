import cv2
import mediapipe as mp
import csv
import os

# 1. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 2. File Setup
DATA_FILE = 'hand_data.csv'

def save_to_csv(label, landmarks):
    file_exists = os.path.isfile(DATA_FILE)
    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        # Flatten the 21 landmarks (x,y,z) into a single row of 63 values + the label
        row = [label] + [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
        writer.writerow(row)

# 3. Camera Loop
cap = cv2.VideoCapture(0)
print("Press '0'-'9' or letters to save data. Press 'q' to quit.")

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Get Key Press
            key = cv2.waitKey(1) & 0xFF
            if ord('a') <= key <= ord('z') or ord('0') <= key <= ord('9'):
                label = chr(key).upper()
                save_to_csv(label, hand_lms.landmark)
                print(f"Saved sample for: {label}")
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow("Hand Collector", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()