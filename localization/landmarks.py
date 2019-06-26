"""A Landmark Class to store landmarks, descriptors and projected keypoints."""


class Landmarks():
    """Landmarks"""

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
        return len(self.landmarks)
