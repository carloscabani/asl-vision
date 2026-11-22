
import cv2
import mediapipe as mp
import time

from pathlib import Path
from multi_hand_extractor import MultiHandExtractor


# Model path
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "hand_landmarker.task"


# MediaPipe configuration
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

extractor = MultiHandExtractor()


with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")

    last_timestamp = 0

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = max(
            int(time.monotonic() * 1000),
            last_timestamp + 1
        )

        last_timestamp = timestamp

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        features = extractor.extract_features(result)

        left_present = features[66] == 1.0
        right_present = features[133] == 1.0

        cv2.putText(
            frame,
            f"Features: {len(features)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Left hand: {'Detected' if left_present else 'Not detected'}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Right hand: {'Detected' if right_present else 'Not detected'}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.imshow("ASL Vision - Multi Hand Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
