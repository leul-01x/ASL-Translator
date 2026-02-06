import os
import cv2
import mediapipe as mp
import csv

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

dataset_size = 500  

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

for idx, letter in enumerate(letters):
    csv_file = os.path.join(DATA_DIR, f'class_{idx}.csv')
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
            writer.writerow(header)

for idx, letter in enumerate(letters):
    print(f'Collecting data for {letter}. Press "C" to start.')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, f'Get ready for {letter}. Press C!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('c'):
            break
        if key == ord('q'):
            print('Quitting early...')
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            exit()

    counter = 0
    while counter < dataset_size:
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

            for i, lm in enumerate(hand.landmark):
                landmarks.extend([lm.x, lm.y, lm.z])
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.putText(frame, f'{i}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            if len(landmarks) == 63:
                with open(os.path.join(DATA_DIR, f'class_{idx}.csv'), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks + [idx])
                counter += 1

        cv2.putText(frame, f'Letter: {letter}  Collected: {counter}/{dataset_size}', (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow('ASL Live Collection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print('Quitting early...')
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            exit()

cap.release()
cv2.destroyAllWindows()

hands.close()

print('Data collection finished!')

print('Data collection finished!')
