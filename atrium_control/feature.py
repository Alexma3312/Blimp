"""This is a feature class."""

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
        # Transform array to list
        self.key_point_list = key_points.tolist()
        self.descriptor_list = descriptors.tolist()

    def create_from_list(self, key_points, descriptors):
        self.key_point_list = key_points
        self.descriptor_list = descriptors
        # Transform list to array
        self.key_points = np.array(self.key_point_list)
        self.descriptors = np.array(self.descriptor_list)

    def get_length(self):
        """ Return the number of key points."""
        assert len(self.key_point_list) == len(self.descriptor_list)
        return len(self.key_point_list)

    def get_keypoint(self, point_index):
        """ Return a key point from array with index point_index."""
        return self.key_points[point_index]

    def get_descriptor(self, descriptor_index):
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