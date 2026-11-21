
import csv
import cv2
import mediapipe as mp

from pathlib import Path
from landmark_extractor import LandmarkExtractor


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
DATASET_PATH = PROJECT_ROOT / "data" / "asl_landmarks.csv"

# Static ASL letters.
# J and Z are excluded because they require movement.
VALID_LETTERS = set("ABCDEFGHIKLMNOPQRSTUVWXY")


def create_dataset():
    """Create the CSV file and its header if it does not exist."""

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        headers = ["label"]

        headers.extend(
            [f"feature_{i}" for i in range(63)]
        )

        with open(
            DATASET_PATH,
            mode="w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(headers)


def save_sample(label, features):
    """Append one labeled hand sample to the CSV file."""

    with open(
        DATASET_PATH,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [label] + features.tolist()
        )


def main():
    create_dataset()

    extractor = LandmarkExtractor()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {MODEL_PATH}"
        )

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    current_label = None
    sample_count = 0
    timestamp = 0

    print("\nASL Vision - Dataset Collector")
    print("--------------------------------")
    print("Press a letter key (A-Y, excluding J).")
    print("Press SPACE to capture a sample.")
    print("Press ESC to quit.\n")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            while camera.isOpened():
                success, frame = camera.read()

                if not success:
                    print("Could not read from webcam.")
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

                timestamp += 33

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp
                )

                features = None

                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0]

                    features = extractor.extract_features(
                        hand_landmarks
                    )

                    # Draw the 21 detected landmarks.
                    height, width, _ = frame.shape

                    for landmark in hand_landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 255, 0),
                            -1
                        )

                    cv2.putText(
                        frame,
                        "Hand detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                else:
                    cv2.putText(
                        frame,
                        "No hand detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                label_text = (
                    current_label
                    if current_label
                    else "Not selected"
                )

                cv2.putText(
                    frame,
                    f"Selected letter: {label_text}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Saved samples: {sample_count}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "SPACE: capture | ESC: quit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "ASL Vision - Dataset Collector",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break

                elif key == ord(" "):
                    if current_label is None:
                        print("Select a letter first.")
                    elif features is None:
                        print("No hand detected. Sample not saved.")
                    else:
                        save_sample(current_label, features)
                        sample_count += 1

                        print(
                            f"Saved sample {sample_count}: "
                            f"Letter {current_label}"
                        )

                elif key != 255:
                    pressed_key = chr(key).upper()

                    if pressed_key in VALID_LETTERS:
                        current_label = pressed_key

                        print(
                            f"Selected letter: {current_label}"
                        )

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("\nDataset collection finished.")
    print(f"Dataset location: {DATASET_PATH}")


if __name__ == "__main__":
    main()
