""" Contains data structures that will be used in mapping and localization."""

import gtsam
import numpy as np
import numpy.testing as npt
from gtsam import Point2, Point3, Pose3


class Features(object):
    """ 
    A Feature Object that stores key points of one image and its corresponding descriptors.

    Args:
        key_points - an N*2 numpy array of N gtsam.Point2 objects
        descriptors - an N*M numpy array of M dimension descriptors, for Superpoint M = 256
                     A descriptor is a M dimensional normalized vector.
        key_point_list - an N*2 list
        descriptor_list - an N*M list
    """

    def __init__(self, key_points=np.zeros((0, 2)), descriptors=np.zeros((0, 256))):
        self.key_points = key_points
        self.descriptors = descriptors

        # Transform Array to list
        self.key_point_list = key_points.tolist()
        self.descriptor_list = descriptors.tolist()

    def get_length(self):
        """ Return the number of key points."""
        assert len(self.key_point_list) == len(self.descriptor_list)
        return len(self.key_point_list)

    def get_keypoint(self, point_index):
        """ Return a key point from array with index point_index."""
        return self.key_points[point_index]

    def descriptor(self, descriptor_index):
        """ Return a decriptor from array with index descriptor_index."""
        return self.descriptors[descriptor_index]

    def get_keypoint_from_list(self, point_index):
        """ Return a key point from list with index point_index."""
        return self.key_point_list[point_index]

    def get_descriptor_from_list(self, descriptor_index):
        """ Return a decriptor from list with index descriptor_index."""
        return self.descriptor_list[descriptor_index]

    def append(self, key_point, descriptor):
        """ Add a key_point with it's descriptor into Feature."""
        self.key_point_list.append(key_point)
        self.descriptor_list.append(descriptor)
        self.key_points = np.array(self.key_point_list)
        self.descriptors = np.array(self.descriptor_list)

    def assert_almost_equal(self, Features):
        """This function is for unittest, to test whether the actual output and expected output are the same

        the size should be the same
        the gtsam.point object should have the same value
        the descriptors should have small L2 distance

        r1 = npt.assert_almost_equal(self.key_points, Features.key_points)
        """
        r2 = npt.assert_almost_equal(self.descriptors, Features.descriptors)
        return r2


class Map(object):
    """
    A Map object that stores landmarks with corresponding descriptors and a camera trajectory.

    Args:
        points - an N*3 numpy array of N gtsam.Point3 objects
        descriptors - an N*M numpy array of M dimension descriptors, for Superpoint M = 256
        landmark_list - an N*3 list
        descriptor_list - an N*M list
        trajectory - an N*6 numpy array of N gtsam.Pose3 objects
    """

    def __init__(self, points=np.zeros((0, 3)), descriptors=np.zeros((0, 256)), trajectory=np.zeros((6, 0))):
        self.landmark_points = points
        self.descriptors = descriptors
        self.landmark_list = points.tolist()
        self.descriptor_list = descriptors.tolist()
        self.trajectory = trajectory

    def get_length(self):
        """ Return the number of key points."""
        assert len(self.landmark_list) == len(self.descriptor_list)
        return len(self.landmark_list)

    def get_landmark(self, point_index):
        """ Return a landmark point from array with index point_index."""
        return self.landmark_points[point_index]

    def get_descriptor(self, descriptor_index):
        """ Return a decriptor from array with index descriptor_index."""
        return self.descriptors[descriptor_index]

    def get_landmark_from_list(self, point_index):
        """ Return a landmark point from list with index point_index."""
        return self.landmark_list[point_index]

    def get_descriptor_from_list(self, descriptor_index):
        """ Return a decriptor from list with index descriptor_index."""
        return self.descriptor_list[descriptor_index]

    def append(self, point, descriptor):
        """ Add a landmark point with it's descriptor into Map."""
        self.landmark_list.append(point)
        self.descriptor_list.append(descriptor)
        self.landmark_points = np.array(self.landmark_list)
        self.descriptors = np.array(self.descriptor_list)

    def add_trajectory(self, trajectory):
        """ Add a trajectory to Map."""
        self.trajectory = trajectory

    def append_trajectory(self, pose):
        """ Add a pose into trajectory."""
        self.trajectory.append(pose)

    def assert_almost_equal(self, Map):
        # This function is for unittest, to test whether the actual output and expected output are the same
        # r1 = npt.assert_almost_equal(self.landmark_points, Map.key_points)
        r2 = npt.assert_almost_equal(self.descriptors, Map.descriptors)
        return r2
