"""A Feature Class to store both keypoints and descriptors."""


class Feature():
    """Feature"""

    def __init__(self, keypoint, descriptor):
        self.keypoint = keypoint
        self.descriptor = descriptor


class Features():
    """Features"""

    def __init__(self, keypoints, descriptors):
        self.keypoints = keypoints
        self.descriptors = descriptors

    def append(self, keypoint, descriptor):
        """Add new feature."""
        self.keypoints.append(keypoint)
        self.descriptors.append(descriptor)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.get_length():
            result = Feature(
                self.keypoints[self.index], self.descriptors[self.index])
            self.index += 1
            return result
        else:
            raise StopIteration

    def __eq__(self, other):
        result = (self.keypoints == other.keypoints) and (
            self.descriptors == other.descriptors)
        return result

    def get_length(self):
        """Return length"""
        assert len(self.keypoints) == len(
            self.descriptors), "Lengths of Keypoints and Descriptors are different."
        return len(self.keypoints)
