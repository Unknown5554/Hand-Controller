
import cv2
import mediapipe as mp
import pyautogui
import time
import math

# ==========================================
# SETTINGS
# ==========================================
MODEL_PATH = "hand_landmarker.task"
CAMERA_ID = 0

PINCH_THRESHOLD = 0.035
RIGHT_CLICK_HOLD = 0.60
DOUBLE_CLICK_WINDOW = 0.30

OPEN_FINGER_RATIO = 1.15
OPEN_HAND_FINGERS_REQUIRED = 4

# Cursor sensitivity
SMOOTHING = 0.45
CURSOR_MARGIN = 0.25
MIRROR_CAMERA = True

# ==========================================
# MOUSE SETUP
# ==========================================
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# ==========================================
# STATE
# ==========================================
smooth_x = SCREEN_W / 2
smooth_y = SCREEN_H / 2

pinching = False
pinch_start = 0.0
right_click_done = False
pending_click_time = None

left_button_held = False
last_timestamp = 0

# ==========================================
# HELPERS
# ==========================================
def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def screen_position(point):
    x = (point.x - CURSOR_MARGIN) / (1 - 2 * CURSOR_MARGIN)
    y = (point.y - CURSOR_MARGIN) / (1 - 2 * CURSOR_MARGIN)

    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))

    return (
        int(x * (SCREEN_W - 1)),
        int(y * (SCREEN_H - 1))
    )


def draw_text(frame, text, y, color=(255, 150, 255)):
    cv2.putText(
        frame, text, (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62, color, 2, cv2.LINE_AA
    )


def is_hand_open(hand):
    wrist = hand[0]

    finger_pairs = [
        (8, 6),    # Index
        (12, 10),  # Middle
        (16, 14),  # Ring
        (20, 18)   # Pinky
    ]

    extended = 0

    for tip_id, joint_id in finger_pairs:
        tip_dist = distance(hand[tip_id], wrist)
        joint_dist = distance(hand[joint_id], wrist)

        if tip_dist > joint_dist * OPEN_FINGER_RATIO:
            extended += 1

    thumb_extended = (
        distance(hand[4], hand[5]) >
        distance(hand[3], hand[5]) * 0.85
    )

    return (
        extended >= OPEN_HAND_FINGERS_REQUIRED
        and thumb_extended
    )


def draw_hand(frame, hand):
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]

    height, width = frame.shape[:2]

    for a, b in connections:
        p1 = (
            int(hand[a].x * width),
            int(hand[a].y * height)
        )
        p2 = (
            int(hand[b].x * width),
            int(hand[b].y * height)
        )
        cv2.line(frame, p1, p2, (255, 100, 0), 2)

    for point in hand:
        cv2.circle(
            frame,
            (int(point.x * width), int(point.y * height)),
            3, (255, 100, 0), -1
        )


def release_left_button():
    global left_button_held

    if left_button_held:
        pyautogui.mouseUp(button="left")
        left_button_held = False


def reset_gestures():
    global pinching, right_click_done, pending_click_time

    pinching = False
    right_click_done = False
    pending_click_time = None


# ==========================================
# CAMERA
# ==========================================
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check CAMERA_ID."
    )

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = max(
                int(time.monotonic() * 1000),
                last_timestamp + 1
            )
            last_timestamp = timestamp

            result = landmarker.detect_for_video(
                mp_image, timestamp
            )

            now = time.monotonic()
            status = "SHOW BOTH HANDS"

            if result.hand_landmarks and len(result.hand_landmarks) == 2:
                hands = result.hand_landmarks

                for hand in hands:
                    draw_hand(frame, hand)

                # Sort by horizontal position in the mirrored image.
                # Left-side hand = gestures.
                # Right-side hand = cursor.
                hands_sorted = sorted(
                    hands,
                    key=lambda hand: sum(p.x for p in hand) / len(hand)
                )

                gesture_hand = hands_sorted[0]
                cursor_hand = hands_sorted[1]

                gesture_center_x = sum(
                    p.x for p in gesture_hand
                ) / len(gesture_hand)

                cursor_center_x = sum(
                    p.x for p in cursor_hand
                ) / len(cursor_hand)

                # Cursor is controlled ONLY by the right-side hand.
                target_x, target_y = screen_position(cursor_hand[8])

                smooth_x += (
                    target_x - smooth_x
                ) * SMOOTHING

                smooth_y += (
                    target_y - smooth_y
                ) * SMOOTHING

                pyautogui.moveTo(
                    int(smooth_x),
                    int(smooth_y)
                )

                # Gestures are controlled ONLY by the left-side hand.
                thumb = gesture_hand[4]
                index = gesture_hand[8]

                pinch_distance = distance(thumb, index)
                is_pinching = pinch_distance < PINCH_THRESHOLD
                open_hand = is_hand_open(gesture_hand)

                if open_hand:
                    # Open palm holds the left mouse button.
                    pinching = False
                    right_click_done = False
                    pending_click_time = None

                    if not left_button_held:
                        pyautogui.mouseDown(button="left")
                        left_button_held = True

                    status = "OPEN PALM: HOLD LEFT CLICK"

                else:
                    # Closing the palm releases left-click.
                    release_left_button()

                    if is_pinching and not pinching:
                        pinching = True
                        pinch_start = now
                        right_click_done = False
                        status = "GESTURE: PINCH"

                    elif is_pinching and pinching:
                        held = now - pinch_start

                        if (
                            held >= RIGHT_CLICK_HOLD
                            and not right_click_done
                        ):
                            pending_click_time = None
                            pyautogui.rightClick()
                            right_click_done = True
                            status = "RIGHT CLICK"

                        elif right_click_done:
                            status = "RIGHT CLICK - RELEASE"

                        else:
                            status = "PINCH..."

                    elif not is_pinching and pinching:
                        held = now - pinch_start
                        pinching = False

                        if right_click_done:
                            status = "READY"

                        elif held < RIGHT_CLICK_HOLD:
                            if (
                                pending_click_time is not None
                                and now - pending_click_time
                                <= DOUBLE_CLICK_WINDOW
                            ):
                                pyautogui.doubleClick(interval=0.08)
                                pending_click_time = None
                                status = "DOUBLE CLICK"

                            else:
                                if pending_click_time is not None:
                                    pyautogui.click()

                                pending_click_time = now
                                status = "CLICK READY"

                    # Complete a pending single-click.
                    if (
                        pending_click_time is not None
                        and not pinching
                        and now - pending_click_time
                        > DOUBLE_CLICK_WINDOW
                    ):
                        pyautogui.click()
                        pending_click_time = None
                        status = "LEFT CLICK"

                    if (
                        not pinching
                        and not right_click_done
                        and status not in (
                            "LEFT CLICK",
                            "DOUBLE CLICK",
                            "CLICK READY"
                        )
                    ):
                        status = "GESTURE HAND READY"

                # Highlight gesture hand fingertips.
                for point in (thumb, index):
                    cv2.circle(
                        frame,
                        (int(point.x * width),
                         int(point.y * height)),
                        9, (255, 0, 200), 2
                    )

                cv2.line(
                    frame,
                    (int(thumb.x * width),
                     int(thumb.y * height)),
                    (int(index.x * width),
                     int(index.y * height)),
                    (255, 0, 200), 2
                )

                cv2.putText(
                    frame,
                    f"Pinch: {pinch_distance:.3f}",
                    (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1
                )

                cv2.putText(
                    frame,
                    "LEFT SIDE: GESTURES",
                    (15, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 150, 255), 2
                )

                cv2.putText(
                    frame,
                    "RIGHT SIDE: CURSOR",
                    (width // 2, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

            else:
                # Both hands are required to keep roles unambiguous.
                release_left_button()
                reset_gestures()
                status = "SHOW BOTH HANDS"

            draw_text(frame, "GESTURE MOUSE V5", 30)
            draw_text(frame, status, 60)
            draw_text(
                frame, "Q / ESC = QUIT",
                90, (255, 255, 255)
            )

            cv2.imshow("Gesture Mouse V5", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

finally:
    try:
        pyautogui.mouseUp(button="left")
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()
