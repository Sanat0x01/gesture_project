import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyautogui
import subprocess
import time

# Load the trained KNN model
with open('gesture_knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_action_time = 0
action_delay = 1  # seconds delay between actions to avoid spamming

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])  # 63 features

            data = np.array(landmark_list).reshape(1, -1)
            prediction = model.predict(data)[0]

            # Display predicted gesture
            cv2.putText(image, f'Gesture: {prediction}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Map gestures to actions with a cooldown delay
            current_time = time.time()
            if current_time - last_action_time > action_delay:
                if prediction == "palm":
                    # Play/Pause media (space key is common)
                    pyautogui.press('space')
                    print("Play/Pause triggered")
                    last_action_time = current_time

                elif prediction == "thumbs_up":
                    # Volume up (usually 'volume up' key, can be simulated)
                    pyautogui.press('volumeup')
                    print("Volume Up triggered")
                    last_action_time = current_time

                elif prediction == "thumbs_down":
                    # Volume down
                    pyautogui.press('volumedown')
                    print("Volume Down triggered")
                    last_action_time = current_time

                elif prediction == "fist":
                    # Switch window (Alt+Tab)
                    subprocess.run(['xdotool', 'key', 'alt+Tab'])
                    print("Switch Window triggered")
                    last_action_time = current_time

            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Recognition", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

