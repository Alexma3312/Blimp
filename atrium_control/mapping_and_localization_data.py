""" Contains data structures that will be used in mapping and localization."""

import gtsam
import numpy as np
from gtsam import Point2, Point3, Pose3


class Features(object):
    """ 
    Store key points of one image and its corresponding descriptors in one Features object

    Args:
        key_points - an N*2 numpy array of N gtsam.Point2 objects
        descriptors - an N*M numpy array of M dimension descriptors, for Superpoint M = 256
    """

    def __init__(self, key_points, descriptors):
        self.key_points = key_points
        self.descriptors = descriptors
        self.key_point_list = key_points.tolist()
        self.descriptor_list = descriptors.tolist()

    def get_keypoint(self, point_index):
        return self.key_points[point_index]

    def descriptor(self, descriptor_index):
        return self.descriptors[descriptor_index]

    def get_keypoint_from_list(self, point_index):
        return self.key_point_list[point_index]

    def get_descriptor_from_list(self, descriptor_index):
        return self.descriptor_list[descriptor_index]



class Map(object):
    """
    Store landmarks and corresponding descriptors in one Map object

    Args:
        points - an N*3 numpy array of N gtsam.Point3 objects
        descriptors - an N*M numpy array of M dimension descriptors, for Superpoint M = 256
    """

    def __init__(self, points, descriptors):
        self.landmark_points = points
        self.descriptors = descriptors
        self.landmark_list = points.tolist()
        self.descriptor_list = descriptors.tolist()

    def get_landmark(self, point_index):
        return self.landmark_points[point_index]

    def get_descriptor(self, descriptor_index):
        return self.descriptors[descriptor_index]

    def get_landmark_from_list(self, point_index):
        return self.landmark_list[point_index]

    def get_descriptor_from_list(self, descriptor_index):
        return self.descriptor_list[descriptor_index]
