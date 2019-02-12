"""Test incremental visual SAM."""

import unittest

import math
import gtsam
from gtsam import Point2, Point3, Pose3
import numpy as np

import cv2

import TrajectoryEstimator

def read_images():
    image_1 = cv2.imread('dataset/wall_data/raw_frame_left.jpg')
    image_2 = cv2.imread('dataset/wall_data/raw_frame_middle.jpg')
    image_3 = cv2.imread('dataset/wall_data/raw_frame_right.jpg')
    return image_1, image_2, image_3

class TestTrajectoryEstimator(unittest.TestCase):

    def setUp(self):
        self.image_1, self.image_2, self.image_3 = read_images()
        self.actual_estimate_trajectory = []
        self.trajectory_estimator = TrajectoryEstimator.TrajectoryEstimator()

    def assertGtsamEquals(self, actual, expected, tol=2):
        """Helper function that prints out actual and expected if not equal."""
        equal = actual.equals(expected, tol)
        if not equal:
            raise self.failureException(
                "Poses are not equal:\n{}!={}".format(actual, expected))

    def assertGtsamListEqual(self, actual_list, expect_list):
        assert len(actual_list) == len(expect_list)
        for i in range(len(actual_list)): 
            self.assertGtsamEquals(actual_list[i], expect_list[i])

    def test_trajectory_estimator(self):
        expect_estimate_trajectory = []
        expect_current_estimate_state = []
        actual_estimate_trajectory, actual_current_estimate_state = self.trajectory_estimator.trajectory_estimator(self.image_1)
        self.actual_estimate_trajectory.append(actual_estimate_trajectory)
        # self.assertGtsamListEqual(self.actual_estimate_trajectory, expect_estimate_trajectory)
        # self.assertGtsamListEqual(self.actual_current_estimate_state, expect_current_estimate_state)
        
        expect_estimate_trajectory = []
        expect_current_estimate_state = []
        actual_estimate_trajectory, actual_current_estimate_state = self.trajectory_estimator.trajectory_estimator(self.image_2)
        self.actual_estimate_trajectory.append(actual_estimate_trajectory)
        # self.assertGtsamListEqual(self.actual_estimate_trajectory, expect_estimate_trajectory)
        # self.assertGtsamListEqual(self.actual_current_estimate_state, expect_current_estimate_state)
        
        expect_estimate_trajectory = []
        expect_current_estimate_state = []
        actual_estimate_trajectory, actual_current_estimate_state = self.trajectory_estimator.trajectory_estimator(self.image_3)
        self.actual_estimate_trajectory.append(actual_estimate_trajectory)
        # self.assertGtsamListEqual(self.actual_estimate_trajectory, expect_estimate_trajectory)
        # self.assertGtsamListEqual(self.actual_current_estimate_state, expect_current_estimate_state)


        self.assertGtsamListEqual([Point3(0,0,0), Point3(2,2,3)],[Point3(0,0,0),Point3(1,1,1)])


if __name__ == "__main__":
    unittest.main()
