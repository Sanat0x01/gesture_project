import cv2
import mediapipe as mp
import csv
import os

# Ask user for gesture name and number of samples
GESTURE_NAME = input("Enter the gesture name: ")
SAMPLES_TO_COLLECT = int(input("Enter number of samples to collect: "))

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)
CSV_FILE = os.path.join(DATA_DIR, 'gesture_data.csv')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print(f"Collecting {SAMPLES_TO_COLLECT} samples for gesture '{GESTURE_NAME}'...")

count = 0
with open(CSV_FILE, mode='a', newline='') as f:
    csv_writer = csv.writer(f)

    while count < SAMPLES_TO_COLLECT:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                csv_writer.writerow([GESTURE_NAME] + landmarks)
                count += 1
                print(f"Sample {count}/{SAMPLES_TO_COLLECT} saved.")

        else:
            cv2.putText(frame, "No Hand Detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Collecting Gestures", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user.")
            break

print(f"Finished collecting {count} samples for gesture '{GESTURE_NAME}'.")
cap.release()
cv2.destroyAllWindows()

