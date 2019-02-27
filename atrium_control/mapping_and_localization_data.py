""" Contains data structures that will be used in mapping and localization."""

import gtsam
import numpy as np
from gtsam import Point2, Point3, Pose3


class Features(object):
    """ 
    Store key points of one image and its corresponding descriptors in one Features object

    Args:
        key_points - a list of N gtsam.Point2 objects
        descriptors - an N*M numpy array of M dimension descriptors, for Superpoint M = 256
    """

    def __init__(self, key_points, descriptors):
        self.key_points = np.array(key_points)
        self.descriptors = descriptors

    def get_keypoint(self, point_index):
        return self.key_points[point_index]

    def get_descriptor(self, descriptor_index):
        return self.descriptors[descriptor_index]


class Map(object):
    """
    Store landmarks and corresponding descriptors in one Map object

    Args:
        points - a list of N gtsam.Point3 objects
        descriptors - an N*M numpy array of M dimension descriptors, for Superpoint M = 256
    """

    def __init__(self, points, descriptors):
        self.landmark_points = np.array(points)
        self.descriptors = descriptors

    def get_landmark(self, point_index):
        return self.landmark_points[point_index]

    def get_descriptor(self, descriptor_index):
        return self.descriptors[descriptor_index]


def create_atrium_map():
    """

    """
    atrium_points = [
        Point3(10.0, -15.0, 15.0),
        Point3(10.0, -5.0, 10.),
        Point3(10.0, 10.0, 5.0),
        Point3(10.0, 15.0, 0.0),
        Point3(10.0, 20.0, -5.0),
        Point3(10.0, -22.0, 15.0),
        Point3(10.0, -10.0, 20.0),
        Point3(10.0, -10.0, 10.0),
        Point3(10.0, 31.0, 17.0),
        Point3(10.0, 35.0, 15.0)]

    atrium_descriptors = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [
        0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [0.7, 0.7, 0.7], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9], [1.1, 1.1, 1.1]])
    
    atrium_map = Map(atrium_points, atrium_descriptors)
    
    return atrium_map


if __name__ == "__main__":
    atrium_map = create_atrium_map()

