
import cv2
import mediapipe as mp
import time
import numpy as np

from landmark_extractor import LandmarkExtractor

from pathlib import Path


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

MAX_HANDS = 2


# ==========================================
# 2. HAND CONNECTIONS
# ==========================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ==========================================
# 3. DRAW HAND LANDMARKS
# ==========================================

def draw_hand_landmarks(frame, hand_landmarks):

    height, width, _ = frame.shape

    points = []

    for landmark in hand_landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 0, 0),
            2
        )


# ==========================================
# 4. INITIALIZE MEDIAPIPE
# ==========================================

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    BaseOptions = mp.tasks.BaseOptions

    HandLandmarker = mp.tasks.vision.HandLandmarker

    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )

    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ======================================
    # 5. INITIALIZE CAMERA
    # ======================================
    extractor = LandmarkExtractor()
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not camera.isOpened():
        print("ERROR: Could not access webcam.")
        return

    print("ASL Vision - Hand Detection Started")
    print("Press Q to exit.")

    # ======================================
    # 6. REAL-TIME DETECTION
    # ======================================

    with HandLandmarker.create_from_options(options) as landmarker:

        while camera.isOpened():

            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read camera frame.")
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = time.monotonic_ns() // 1_000_000

            # Detect hands
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            detected_hands = len(result.hand_landmarks)

            # Draw landmarks
            for index, hand_landmarks in enumerate(
                result.hand_landmarks
            ):

                draw_hand_landmarks(
                    frame,
                    hand_landmarks
                )

                handedness = result.handedness[index][0]

                hand_label = handedness.category_name

                cv2.putText(
                    frame,
                    hand_label,
                    (30, 60 + index * 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


                # Extract features from the detected hand
                features = extractor.extract_features(hand_landmarks)
                print(
                    f"Detected hand features: {features.shape}"
                )

            # Display detection information
            status = f"Hands detected: {detected_hands}"

            cv2.putText(
                frame,
                status,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "ASL Vision - Hand Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ======================================
    # 7. RELEASE RESOURCES
    # ======================================

    camera.release()
    cv2.destroyAllWindows()

    print("ASL Vision stopped successfully.")


if __name__ == "__main__":
    main()
