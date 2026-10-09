
import cv2
import mediapipe as mp
import numpy as np
import math
import time

MODEL_PATH = "hand_landmarker.task"
CAMERA_ID = 0

# Colors use BGR format
VIOLET = (255, 0, 200)
LIGHT_VIOLET = (255, 150, 255)
BLUE = (255, 100, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (35, 35, 35)

# Appearance
LINE_THICKNESS = 2
HAND_LINE_THICKNESS = 1
DOT_RADIUS = 4
BOX_OPACITY = 0.65

# Timing and tracking
TOUCH_THRESHOLD = 0.035
CONNECT_THRESHOLD = 0.055
PINCH_HOLD_SECONDS = 0.5
CONNECT_HOLD_SECONDS = 0.5
MAX_HANDS = 2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

camera = cv2.VideoCapture(CAMERA_ID)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

hand_states = {}
box_on = False
connect_start = None
connect_latched = False
timestamp_ms = 0


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def pixel_point(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


try:
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms += 33
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            current_time = time.time()

            hands = {}

            # Detect and draw both hands
            for hand_index, landmarks in enumerate(result.hand_landmarks):
                label = f"hand_{hand_index}"

                if (
                    result.handedness
                    and hand_index < len(result.handedness)
                    and result.handedness[hand_index]
                ):
                    label = result.handedness[hand_index][0].category_name

                state = hand_states.setdefault(label, {
                    "line_on": False,
                    "pinch_start": None,
                    "pinch_latched": False
                })

                hands[label] = landmarks

                # Black hand skeleton lines
                for start, end in HAND_CONNECTIONS:
                    p1 = pixel_point(landmarks[start], width, height)
                    p2 = pixel_point(landmarks[end], width, height)
                    cv2.line(
                        frame, p1, p2,
                        BLACK, int(HAND_LINE_THICKNESS), cv2.LINE_AA
                    )

                # Blue hand landmark dots
                for lm in landmarks:
                    cv2.circle(
                        frame,
                        pixel_point(lm, width, height),
                        int(DOT_RADIUS),
                        BLUE,
                        -1,
                        cv2.LINE_AA
                    )

                thumb = landmarks[4]
                index = landmarks[8]

                thumb_p = pixel_point(thumb, width, height)
                index_p = pixel_point(index, width, height)

                # White fingertip markers
                cv2.circle(frame, thumb_p, 6, WHITE, -1, cv2.LINE_AA)
                cv2.circle(frame, index_p, 6, WHITE, -1, cv2.LINE_AA)

                # Hold pinch for 0.5 seconds to toggle the hand's line
                pinching = distance(thumb, index) < TOUCH_THRESHOLD

                if pinching:
                    if state["pinch_start"] is None:
                        state["pinch_start"] = current_time

                    held = current_time - state["pinch_start"]

                    if (
                        held >= PINCH_HOLD_SECONDS
                        and not state["pinch_latched"]
                    ):
                        state["line_on"] = not state["line_on"]
                        state["pinch_latched"] = True

                        # Intentional line OFF disables the box
                        if not state["line_on"]:
                            box_on = False
                            connect_start = None
                            connect_latched = False
                else:
                    state["pinch_start"] = None
                    state["pinch_latched"] = False
                    held = 0

                # Individual violet line
                if state["line_on"]:
                    cv2.line(
                        frame,
                        thumb_p,
                        index_p,
                        VIOLET,
                        int(LINE_THICKNESS),
                        cv2.LINE_AA
                    )

                # Pinch progress bar
                if pinching and state["pinch_start"] is not None:
                    progress = min(
                        1.0,
                        (current_time - state["pinch_start"])
                        / PINCH_HOLD_SECONDS
                    )

                    bar_x = 20 if hand_index == 0 else width - 220

                    cv2.rectangle(
                        frame,
                        (bar_x, 30),
                        (bar_x + 200, 45),
                        GRAY,
                        -1
                    )

                    cv2.rectangle(
                        frame,
                        (bar_x, 30),
                        (bar_x + int(200 * progress), 45),
                        VIOLET,
                        -1
                    )

                status = "LINE ON" if state["line_on"] else "LINE OFF"

                cv2.putText(
                    frame,
                    f"{label}: {status}",
                    (20, 75 + hand_index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    VIOLET if state["line_on"] else WHITE,
                    2
                )

            # Preserve line and box states through tracking glitches
            for label in list(hand_states):
                if label not in hands:
                    hand_states[label]["pinch_start"] = None
                    hand_states[label]["pinch_latched"] = False

            # Box logic
            if len(hands) == 2:
                hand_items = list(hands.items())
                label_a, hand_a = hand_items[0]
                label_b, hand_b = hand_items[1]

                both_lines_on = (
                    hand_states[label_a]["line_on"]
                    and hand_states[label_b]["line_on"]
                )

                if not both_lines_on:
                    box_on = False
                    connect_start = None
                    connect_latched = False

                else:
                    thumb_a = hand_a[4]
                    index_a = hand_a[8]
                    thumb_b = hand_b[4]
                    index_b = hand_b[8]

                    thumbs_touching = (
                        distance(thumb_a, thumb_b) < CONNECT_THRESHOLD
                    )
                    indexes_touching = (
                        distance(index_a, index_b) < CONNECT_THRESHOLD
                    )

                    both_pairs_touching = (
                        thumbs_touching and indexes_touching
                    )

                    if both_pairs_touching:
                        if connect_start is None:
                            connect_start = current_time

                        if (
                            current_time - connect_start
                            >= CONNECT_HOLD_SECONDS
                            and not connect_latched
                        ):
                            box_on = not box_on
                            connect_latched = True
                    else:
                        connect_start = None
                        connect_latched = False

                    # Black-filled box with violet border
                    if box_on:
                        thumb_a_p = pixel_point(
                            thumb_a, width, height
                        )
                        index_a_p = pixel_point(
                            index_a, width, height
                        )
                        thumb_b_p = pixel_point(
                            thumb_b, width, height
                        )
                        index_b_p = pixel_point(
                            index_b, width, height
                        )

                        polygon = np.array([
                            thumb_a_p,
                            index_a_p,
                            index_b_p,
                            thumb_b_p
                        ], dtype=np.int32)

                        # Blend transparent black fill
                        overlay = frame.copy()

                        cv2.fillPoly(
                            overlay,
                            [polygon],
                            BLACK,
                            lineType=cv2.LINE_AA
                        )

                        frame = cv2.addWeighted(
                            overlay,
                            BOX_OPACITY,
                            frame,
                            1.0 - BOX_OPACITY,
                            0
                        )

                        # Violet border
                        cv2.polylines(
                            frame,
                            [polygon],
                            True,
                            VIOLET,
                            int(LINE_THICKNESS),
                            cv2.LINE_AA
                        )

                        # Violet corner markers
                        for point in [
                            thumb_a_p,
                            index_a_p,
                            index_b_p,
                            thumb_b_p
                        ]:
                            cv2.circle(
                                frame,
                                point,
                                6,
                                LIGHT_VIOLET,
                                -1,
                                cv2.LINE_AA
                            )

                status = "BOX ON" if box_on else "BOX OFF"

                cv2.putText(
                    frame,
                    f"{status} - touch both pairs to toggle",
                    (20, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    VIOLET if box_on else LIGHT_VIOLET,
                    2
                )

            else:
                # Keep the box state when tracking temporarily fails
                cv2.putText(
                    frame,
                    "Waiting for both hands...",
                    (20, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    LIGHT_VIOLET,
                    2
                )

            cv2.imshow("Violet Hand Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
