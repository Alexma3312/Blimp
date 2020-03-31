"""A Feature Class to store both keypoints and descriptors."""


class Feature():
    """Feature"""

    def __init__(self, keypoint, descriptor):
        self.keypoint = keypoint
        self.descriptor = descriptor


class Features():
    """Features\n
    Parameters: \n
        keypoints: Nx2 array
        descriptors: Nx256 array
    """

    def __init__(self, keypoints, descriptors):
        self.keypoints = keypoints
        self.descriptors = descriptors

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

    def get_length(self):
        """Return length"""
        assert self.keypoints.shape[0] == self.descriptors.shape[0], "Lengths of Keypoints and Descriptors are different."
        return self.keypoints.shape[0]
