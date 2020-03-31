""" This is a Map class."""

import gtsam
import numpy as np
import numpy.testing as npt
from gtsam import Point2, Point3, Pose3

class Map(object):
    """
    A Map object that stores landmarks with corresponding descriptors and a camera trajectory.

    Args:
        points - an N*3 numpy array of N gtsam.Point3 objects
        descriptors - an N*M numpy array of M dimension descriptors, for Superpoint M = 256
        landmark_list - an N*3 list
        descriptor_list - an N*M list
        trajectory - an N*6 list of N gtsam.Pose3 objects
    """

    def __init__(self, points=np.zeros((0, 3)), descriptors=np.zeros((0, 256)), trajectory=[]):
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
