
import csv
import cv2
import mediapipe as mp
import uuid

from pathlib import Path
from landmark_extractor import LandmarkExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
DATASET_PATH = PROJECT_ROOT / "data" / "asl_dynamic_landmarks.csv"

SEQUENCE_LENGTH = 30
VALID_LETTERS = {"J", "Z"}
FEATURE_COUNT = 63


def create_dataset():
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        headers = ["sequence_id", "label", "frame_index"]
        headers.extend(
            [f"feature_{i}" for i in range(FEATURE_COUNT)]
        )

        with open(DATASET_PATH, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)


def save_sequence(label, features_sequence):
    sequence_id = str(uuid.uuid4())

    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        for frame_index, features in enumerate(features_sequence):
            writer.writerow([
                sequence_id,
                label,
                frame_index,
                *features.tolist()
            ])

    print(f"Saved sequence: {label} | ID: {sequence_id}")


def main():
    create_dataset()

    extractor = LandmarkExtractor()

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
        raise RuntimeError("Could not open webcam.")

    current_label = None
    recording = False
    features_sequence = []
    timestamp = 0

    print("\nASL Vision - Dynamic Dataset Collector")
    print("---------------------------------------")
    print("Press J or Z to select a letter.")
    print("Press SPACE to record a sequence.")
    print("Press ESC to quit.\n")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            while camera.isOpened():
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

                # Collect frames while recording.
                if recording:
                    if features is None:
                        print("Hand lost. Sequence cancelled.")
                        recording = False
                        features_sequence = []

                    else:
                        features_sequence.append(features)

                        if len(features_sequence) >= SEQUENCE_LENGTH:
                            save_sequence(
                                current_label,
                                features_sequence
                            )

                            recording = False
                            features_sequence = []

                label_text = current_label or "Not selected"

                cv2.putText(
                    frame,
                    f"Selected letter: {label_text}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                if recording:
                    status = (
                        f"Recording: "
                        f"{len(features_sequence)}/{SEQUENCE_LENGTH}"
                    )
                else:
                    status = "Ready"

                cv2.putText(
                    frame,
                    status,
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "SPACE: record | ESC: quit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "ASL Vision - Dynamic Dataset Collector",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break

                elif not recording and key == ord(" "):
                    if current_label is None:
                        print("Select J or Z first.")

                    elif features is None:
                        print("No hand detected. Cannot start.")

                    else:
                        recording = True
                        features_sequence = []
                        print(f"Recording letter {current_label}...")

                elif not recording and key in (ord("j"), ord("z")):
                    current_label = chr(key).upper()
                    print(f"Selected letter: {current_label}")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("\nDynamic dataset collection finished.")
    print(f"Dataset location: {DATASET_PATH}")


if __name__ == "__main__":
    main()
