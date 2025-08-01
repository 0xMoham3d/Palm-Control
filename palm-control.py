import cv2
import mediapipe as mp
import pyautogui
import time
import platform
import sys

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    sys.exit(1)

prev_gesture = None
last_action_time = 0
current_os = platform.system().lower()

def count_fingers(hand_landmarks):
    fingers_up = 0
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers_up += 1
    for tip_id in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]:
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[tip_id - 2]
        if tip.y < pip.y:
            fingers_up += 1
    return fingers_up

print("Show an open palm to play/resume, and a fist (no fingers) to pause.")
print(f"Running on: {current_os}")

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Warning: Frame capture failed. Retrying...")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result or not result.multi_hand_landmarks:
        cv2.imshow("Hand Gesture Controller", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for hand_landmarks in result.multi_hand_landmarks:
        fingers = count_fingers(hand_landmarks)
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        current_time = time.time()

        if fingers == 0 and prev_gesture != 'pause' and current_time - last_action_time > 1:
            print("PAUSE")
            pyautogui.press('k')
            prev_gesture = 'pause'
            last_action_time = current_time

        elif fingers >= 4 and prev_gesture != 'play' and current_time - last_action_time > 1:
            print("PLAY")
            pyautogui.press('k')
            prev_gesture = 'play'
            last_action_time = current_time

    cv2.imshow("Hand Gesture Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

