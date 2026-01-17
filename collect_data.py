import os
import cv2
import mediapipe as mp
import csv


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 250   
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

current_class = 0  

while True:
    print(f'Current class {current_class}. Press C to start recording, Q to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, 'Press C to record, Q to quit', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('c'):
            break
        elif key == ord('q'):
            print("Window closed by Q")
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            exit()

    # Prepare CSV for this class
    class_file = os.path.join(DATA_DIR, f'class_{current_class}.csv')
    if not os.path.exists(class_file):
        with open(class_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f'x{i}' for i in range(21)] + \
                     [f'y{i}' for i in range(21)] + \
                     [f'z{i}' for i in range(21)] + ['label']
            writer.writerow(header)

    # Collect samples
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            with open(class_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(landmarks + [current_class])

            counter += 1
            cv2.putText(frame, f'Collected: {counter}/{dataset_size}', (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            print("Window closed by Q")
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            exit()

    print(f"Finished class {current_class}. Press C to continue or Q to quit.")
    current_class += 1 
    