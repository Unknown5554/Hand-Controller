
import os
import sys
import time
import math
import tkinter as tk
from tkinter import messagebox

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

APP_NAME = "Gesture Mouse"
MODEL_NAME = "hand_landmarker.task"

MIRROR_CAMERA = True
SMOOTHING = 0.45
CURSOR_MARGIN = 0.25

PINCH_CLICK_MAX_SECONDS = 0.35
DOUBLE_CLICK_SECONDS = 0.35
RIGHT_CLICK_HOLD_SECONDS = 0.60

# More forgiving pinch thresholds
PINCH_DISTANCE_RATIO = 0.30
PINCH_RELEASE_RATIO = 0.38

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01


def resource_path(filename):
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def ask_permission():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    allowed = messagebox.askyesno(
        APP_NAME,
        "Allow Gesture Mouse to use your webcam and control your mouse?\n\n"
        "Hand tracking runs locally on this PC.\n"
        "Gestures can move the cursor and click.\n\n"
        "Start Gesture Mouse?",
        icon="warning",
    )
    root.destroy()
    return allowed


def show_error(message):
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(APP_NAME, message)
    root.destroy()


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_is_up(hand, tip, pip):
    return hand[tip].y < hand[pip].y


def is_open_palm(hand):
    fingers = (
        finger_is_up(hand, 8, 6)
        and finger_is_up(hand, 12, 10)
        and finger_is_up(hand, 16, 14)
        and finger_is_up(hand, 20, 18)
    )
    thumb_extended = distance(hand[4], hand[5]) > distance(hand[3], hand[5]) * 1.05
    return fingers and thumb_extended


def hand_center_x(hand):
    return sum(hand[i].x for i in (0, 5, 9, 13, 17)) / 5.0


def draw_hand_landmarks(frame, hands):
    """Draw hand landmarks and connecting lines."""
    height, width = frame.shape[:2]
    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    for hand in hands:
        points = []
        for landmark in hand:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        for connection in connections:
            start = connection.start
            end = connection.end
            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        for index, point in enumerate(points):
            color = (0, 0, 255) if index in (4, 8) else (255, 180, 0)
            cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)


def main():
    if not ask_permission():
        return

    model_path = resource_path(MODEL_NAME)
    if not os.path.isfile(model_path):
        show_error(f"Cannot find {MODEL_NAME} beside the script.")
        return

    cap = None
    left_button_held = False

    try:
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.50,
        )

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            show_error("Could not open the webcam. Check camera permissions.")
            return

        screen_w, screen_h = pyautogui.size()
        last_x, last_y = pyautogui.position()

        pinch_was_down = False
        pinch_started = 0.0
        right_click_fired = False
        pending_click_time = None
        last_timestamp_ms = 0

        fps_start = time.perf_counter()
        frame_count = 0
        fps = 0.0

        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if MIRROR_CAMERA:
                    frame = cv2.flip(frame, 1)

                now = time.perf_counter()
                frame_count += 1
                if now - fps_start >= 1.0:
                    fps = frame_count / (now - fps_start)
                    frame_count = 0
                    fps_start = now

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                timestamp_ms = int(now * 1000)
                timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                hands = result.hand_landmarks or []

                # Restore the landmark skeleton lines.
                draw_hand_landmarks(frame, hands)

                pinch_text = "Pinch: waiting for gesture"

                if len(hands) >= 2:
                    hands = sorted(hands, key=hand_center_x)
                    gesture_hand = hands[0]
                    cursor_hand = hands[-1]

                    usable = 1.0 - 2.0 * CURSOR_MARGIN
                    nx = max(0.0, min(1.0, (cursor_hand[8].x - CURSOR_MARGIN) / usable))
                    ny = max(0.0, min(1.0, (cursor_hand[8].y - CURSOR_MARGIN) / usable))

                    target_x = nx * screen_w
                    target_y = ny * screen_h
                    last_x += (target_x - last_x) * SMOOTHING
                    last_y += (target_y - last_y) * SMOOTHING
                    pyautogui.moveTo(int(last_x), int(last_y), _pause=False)

                    palm_width = max(
                        distance(gesture_hand[5], gesture_hand[17]), 0.001
                    )
                    pinch_ratio = distance(gesture_hand[4], gesture_hand[8]) / palm_width

                    threshold = PINCH_RELEASE_RATIO if pinch_was_down else PINCH_DISTANCE_RATIO
                    pinch_down = pinch_ratio < threshold

                    pinch_text = f"Pinch ratio: {pinch_ratio:.2f} | {'PINCHED' if pinch_down else 'OPEN'}"

                    if pinch_down and not pinch_was_down:
                        pinch_started = now
                        right_click_fired = False

                    if pinch_down and not right_click_fired:
                        if now - pinch_started >= RIGHT_CLICK_HOLD_SECONDS:
                            pyautogui.click(button="right")
                            right_click_fired = True
                            pinch_text = "RIGHT CLICK!"

                    if not pinch_down and pinch_was_down:
                        held = now - pinch_started
                        if not right_click_fired and held <= PINCH_CLICK_MAX_SECONDS:
                            if (
                                pending_click_time is not None
                                and now - pending_click_time <= DOUBLE_CLICK_SECONDS
                            ):
                                pyautogui.doubleClick()
                                pending_click_time = None
                                pinch_text = "DOUBLE CLICK!"
                            else:
                                pending_click_time = now

                        right_click_fired = False

                    if (
                        pending_click_time is not None
                        and now - pending_click_time > DOUBLE_CLICK_SECONDS
                    ):
                        pyautogui.click()
                        pending_click_time = None
                        pinch_text = "LEFT CLICK!"

                    pinch_was_down = pinch_down

                    if is_open_palm(gesture_hand) and not left_button_held:
                        pyautogui.mouseDown(button="left")
                        left_button_held = True
                    elif not is_open_palm(gesture_hand) and left_button_held:
                        pyautogui.mouseUp(button="left")
                        left_button_held = False

                    status = "Left-side hand: gestures | Right-side hand: cursor"

                else:
                    if left_button_held:
                        pyautogui.mouseUp(button="left")
                        left_button_held = False

                    pinch_was_down = False
                    right_click_fired = False

                    if (
                        pending_click_time is not None
                        and now - pending_click_time > DOUBLE_CLICK_SECONDS
                    ):
                        pyautogui.click()
                        pending_click_time = None

                    status = "Show both hands to the camera"

                cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (40, 240, 80), 2, cv2.LINE_AA)
                cv2.putText(frame, pinch_text, (12, 54), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"FPS: {fps:.0f} | Q / ESC: quit", (12, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 240, 80), 2, cv2.LINE_AA)

                cv2.imshow("Gesture Mouse - Q or ESC to quit", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except pyautogui.FailSafeException:
        show_error("Emergency stop triggered.")
    except Exception as exc:
        show_error(f"Gesture Mouse stopped because of an error:\n\n{exc}")
    finally:
        if left_button_held:
            try:
                pyautogui.mouseUp(button="left")
            except Exception:
                pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
