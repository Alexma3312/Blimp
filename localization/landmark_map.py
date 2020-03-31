"""Landmark Map"""


class LandmarkMap():
    """Landmark Map\n
    Parameters:\n
        landmark points: Nx3 array
        descriptors: Nx256 array
    """
    def __init__(self, landmark_points, descriptors):
        self.landmarks = landmark_points
        self.descriptors = descriptors

    def landmark(self, index):
        """Get landmark with index
        Return:
            landmark: (2,) array
        """

        return self.landmarks[index, :]

    def descriptor(self, index):
        """Get descriptor with index
        Return:
            descriptor: (256,) array
        """

        return self.descriptors[index, :]

    def get_length(self):
        """Return length"""
        assert self.landmarks.shape[0] == self.descriptors.shape[0], "Lengths of Keypoints and Descriptors are different."
        return self.landmarks.shape[0]