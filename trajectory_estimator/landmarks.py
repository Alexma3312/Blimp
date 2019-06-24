"""A Landmark Class to store landmarks, descriptors and projected keypoints."""

class Landmarks():
    """Landmarks"""
    def __init__(self, landmarks, descriptors):
        self.landmarks = landmarks
        self.descriptors = descriptors
        self.keypoints = []
        assert len(landmarks) == len(
            descriptors), "Lengths of Landmarks and Descriptors are different."
        self.length = len(landmarks)

    def append(self, landmark, keypoint, descriptor):
        """Add new feature."""
        self.landmarks.append(landmark)
        self.keypoints.append(keypoint)
        self.descriptors.append(descriptor)
    
    def load_keypoints(self, keypoints):
        self.keypoints = keypoints
