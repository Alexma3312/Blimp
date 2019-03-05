"""Test trajectory estimator."""

import unittest

import math
import gtsam
from gtsam import Point2, Point3, Pose3, symbol
import numpy as np
from sfm import sfm_data

import cv2

from atrium_control.trajectory_estimator import read_image, TrajectoryEstimator
from tests.trajectory_estimator_data import create_atrium_map, Trajectory

def read_images(width, height):
    """
    Read a list of images from the dataset folder then resize and transfer each image to grey scale image 
    """
    image_list = []
    image_1 = read_image('datasets/wall_corresponding_feature_data/raw_frame_left.jpg', [width,height])
    image_2 = read_image('datasets/wall_corresponding_feature_data/raw_frame_middle.jpg', [width,height])
    image_3 = read_image('datasets/wall_corresponding_feature_data/raw_frame_right.jpg', [width,height])
    image_list.append(image_1)
    image_list.append(image_2)
    image_list.append(image_3)
    return image_list


class TestTrajectoryEstimator(unittest.TestCase):

    def setUp(self):

        # Import SFM map output as Trajectory Estimator map input
        self.atrium_map = create_atrium_map()

        # Create an input image list
        self.image_list = read_images(640,480)

        # Use the poses in SFM as the actual estimate trajectory
        #self.past_trajectory_estimate = SFMdata.createPoses()
        self.trajectory_estimator = TrajectoryEstimator(self.atrium_map, 4, 4, 0.7, 2)
        self.trajectory = Trajectory(128,640,480,5)
        self.trajectory.create_poses()
        self.trajectory.create_past_poses(0.01)  
        self.trajectory.create_project_features()
        self.trajectory.create_map_indices()
        self.trajectory.create_superpoint_features()
        self.trajectory.create_matched_features()
        self.trajectory.create_visible_map()
        

    def assertGtsamEquals(self, actual, expected, tol=1e-2):
        """Helper function that prints out actual and expected if not equal."""
        equal = actual.equals(expected, tol)
        if not equal:
            raise self.failureException(
                "Poses are not equal:\n{}!={}".format(actual, expected))

    def assertGtsamListEqual(self, actual_list, expect_list):
        assert len(actual_list) == len(expect_list)
        for i in range(len(actual_list)):
            self.assertGtsamEquals(actual_list[i], expect_list[i])

    def test_landmarks_projection(self):
        for pose in self.trajectory.past_poses:
            project_features, map_indices = self.trajectory_estimator.landmarks_projection(pose)

    def test_descriptor_match(self):
        # Calculate distances of unit normalized vectors
        desc1 = np.array([0,0,1])
        desc2 = np.array([0,0,1])
        dmat = np.dot(desc1, desc2.T)
        dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
        self.assertEqual(dmat, 0)

    def test_data_association(self):
        for i,pose in enumerate(self.trajectory.past_poses):
            matched_features, visible_map = self.trajectory_estimator.data_association(self.trajectory.superpoint_features[i],self.trajectory.projected_features[i],self.trajectory.map_indices[i])

    def test_trajectory_estimator(self):
        for i,pose in enumerate(self.trajectory.past_poses):
            result = self.trajectory_estimator.trajectory_estimator(self.trajectory.matched_features[i], self.trajectory.visible_map[i],pose)
            pose_i = result.atPose3(symbol(ord('x'), 0))
            self.assertGtsamEquals(pose_i, self.trajectory.poses[i],0.5)
if __name__ == "__main__":
    unittest.main()
