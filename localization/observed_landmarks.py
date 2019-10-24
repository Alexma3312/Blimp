"""A Landmark Class to store landmarks, descriptors and projected keypoints."""


class Landmark():
    """Landmark"""
    def __init__(self, landmark_point, keypoints, descriptors, mean_descriptor):
        self.landmark_point = landmark_point
        self.associate_keypoints = keypoints
        self.associate_descriptors = descriptors
        self.descriptor = mean_descriptor

    def calculate_mean_descriptor(self):
        """"""
        pass


class ObservedLandmarks():
    """Observe Landmarks"""

    def __init__(self, landmarks, descriptors):
        self.landmarks = landmarks
        self.descriptors = descriptors
        self.keypoints = []

    def append(self, landmark, descriptor, keypoint):
        """Add new feature."""
        self.landmarks.append(landmark)
        self.keypoints.append(keypoint)
        self.descriptors.append(descriptor)

    def load_keypoints(self, keypoints):
        """Load keypoints"""
        self.keypoints = keypoints

    def __eq__(self, other):
        result = (self.landmarks == other.landmarks) and (
            self.descriptors == other.descriptors)
        return result

    def get_length(self):
        """Return length"""
        assert len(self.landmarks) == len(
            self.descriptors), "Lengths of Landmarks and Descriptors are different."
        if self.landmarks:
            return len(self.landmarks)
        return 0
