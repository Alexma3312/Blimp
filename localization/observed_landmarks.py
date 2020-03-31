"""A Landmark Class to store landmarks, descriptors and projected keypoints."""
import numpy as np


class ObservedLandmarks():
    """Observe Landmarks\n
    Parameters:\n
        landmarks: Nx3 array
        descriptors: Nx256 array
        keypoints: Nx2 array
    """

    def __init__(self):
        self.landmarks = np.empty((0,3))
        self.descriptors = np.empty((0,256))
        self.keypoints = np.empty((0,2))

    def keypoint(self, index):
        """Get keypoint with index
        Return:
            keypoint: (2,) array
        """

        return self.keypoints[index, :]

    def descriptor(self, index):
        """Get descriptor with index
        Return:
            descriptor: (256,) array
        """

        return self.descriptors[index, :]

    def landmark(self, index):
        """Get landmark with index
        Return:
            landmark: (3,) array
        """

        return self.landmarks[index, :]

    def append(self, landmark, descriptor, keypoint):
        """Add new feature.
        Parameters:
            landmark: 1x3 array
            descriptor: 1x256 array
            keypoint: 1x2 array
        """
        self.landmarks = np.vstack((self.landmarks, landmark))
        self.keypoints = np.vstack((self.keypoints, keypoint))
        self.descriptors = np.vstack((self.descriptors, descriptor))

    # def load_keypoints(self, keypoints):
    #     """Load keypoints"""
    #     self.keypoints = keypoints

    # def __eq__(self, other):
    #     result = (self.landmarks == other.landmarks) and (
    #         self.descriptors == other.descriptors)
    #     return result

    def get_length(self):
        """Return length"""
        assert self.keypoints.shape[0] == self.descriptors.shape[0], "Lengths of Keypoints and Descriptors are different."
        return self.keypoints.shape[0]
