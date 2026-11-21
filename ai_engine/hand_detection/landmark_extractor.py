
import numpy as np


class LandmarkExtractor:
    """
    Extracts and normalizes the 21 hand landmarks
    detected by MediaPipe.
    """

    LANDMARK_COUNT = 21
    COORDINATES_PER_LANDMARK = 3

    def extract_coordinates(self, hand_landmarks):
        """
        Extract x, y and z coordinates from
        MediaPipe hand landmarks.
        """

        coordinates = []

        for landmark in hand_landmarks:
            coordinates.append([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        return np.array(coordinates, dtype=np.float32)

    def normalize_coordinates(self, coordinates):
        """
        Normalize landmarks relative to the wrist
        and the hand's overall scale.
        """

        coordinates = coordinates.copy()

        # Landmark 0 represents the wrist
        wrist = coordinates[0].copy()

        # Translate all points relative to wrist
        coordinates -= wrist

        # Calculate hand scale using the maximum
        # distance from the wrist
        distances = np.linalg.norm(
            coordinates,
            axis=1
        )

        scale = np.max(distances)

        if scale > 0:
            coordinates /= scale

        return coordinates

    def extract_features(self, hand_landmarks):
        """
        Return a normalized feature vector
        containing 63 numerical values.
        """

        coordinates = self.extract_coordinates(
            hand_landmarks
        )

        normalized = self.normalize_coordinates(
            coordinates
        )

        return normalized.flatten()
