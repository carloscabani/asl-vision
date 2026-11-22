
import numpy as np

from landmark_extractor import LandmarkExtractor


class MultiHandExtractor:
    """
    Extracts features from both hands using MediaPipe.
    Each hand contains 67 features:
    63 normalized coordinates,
    3 wrist position coordinates,
    1 presence indicator.
    """

    FEATURES_PER_HAND = 67
    TOTAL_FEATURES = 134

    def __init__(self):
        self.hand_extractor = LandmarkExtractor()

    def empty_hand_features(self):
        """
        Return an empty feature block for a missing hand.
        """

        return np.zeros(
            self.FEATURES_PER_HAND,
            dtype=np.float32
        )

    def extract_hand_features(self, hand_landmarks):
        """
        Extract normalized hand features and
        preserve the original wrist position.
        """

        normalized_features = (
            self.hand_extractor.extract_features(
                hand_landmarks
            )
        )

        wrist_position = np.array([
            hand_landmarks[0].x,
            hand_landmarks[0].y,
            hand_landmarks[0].z
        ], dtype=np.float32)

        presence = np.array(
            [1.0],
            dtype=np.float32
        )

        return np.concatenate([
            normalized_features,
            wrist_position,
            presence
        ])

    def extract_features(self, detection_result):
        """
        Return a fixed-size feature vector
        containing left and right hand information.
        """

        left_hand = self.empty_hand_features()
        right_hand = self.empty_hand_features()

        if not detection_result.hand_landmarks:
            return np.concatenate([
                left_hand,
                right_hand
            ])

        for index, hand_landmarks in enumerate(
            detection_result.hand_landmarks
        ):

            handedness = (
                detection_result.handedness[index][0]
            )

            label = handedness.category_name.lower()

            # Correct handedness for mirrored webcam input
            if label == "left":
                label = "right"
            else:
                label = "left"

            features = self.extract_hand_features(
                hand_landmarks
            )

            if label == "left":
                left_hand = features

            elif label == "right":
                right_hand = features

        return np.concatenate([
            left_hand,
            right_hand
        ])
