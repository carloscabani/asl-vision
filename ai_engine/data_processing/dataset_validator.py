
import csv
import math

from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "asl_landmarks.csv"

FEATURE_COUNT = 63

VALID_LETTERS = set("ABCDEFGHIKLMNOPQRSTUVWXY")


def validate_dataset():

    print("\nASL Vision - Dataset Validator")
    print("--------------------------------")

    if not DATASET_PATH.exists():
        print("ERROR: Dataset file not found.")
        return

    sample_counts = Counter()
    errors = []

    total_samples = 0

    expected_headers = (
        ["label"] +
        [f"feature_{i}" for i in range(FEATURE_COUNT)]
    )

    with open(
        DATASET_PATH,
        mode="r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != expected_headers:
            print("ERROR: Dataset headers are incorrect.")
            return

        for row_number, row in enumerate(reader, start=2):

            total_samples += 1

            label = row.get("label", "")

            if label not in VALID_LETTERS:
                errors.append(
                    f"Row {row_number}: Invalid label '{label}'."
                )
                continue

            sample_counts[label] += 1

            try:
                features = [
                    float(row[f"feature_{i}"])
                    for i in range(FEATURE_COUNT)
                ]

                if not all(math.isfinite(value) for value in features):
                    raise ValueError("Non-finite value detected.")

            except (ValueError, TypeError, KeyError):
                errors.append(
                    f"Row {row_number}: Invalid feature values."
                )

    print(f"\nTotal samples: {total_samples}")

    print("\nSamples per letter:")

    for letter, count in sorted(sample_counts.items()):
        print(f"Letter {letter}: {count} samples")

    print("\nValidation results:")

    if errors:
        print(f"Found {len(errors)} error(s):")

        for error in errors:
            print(error)

    else:
        print("All samples passed validation.")

    print("\nDataset validation finished.")


if __name__ == "__main__":
    validate_dataset()
