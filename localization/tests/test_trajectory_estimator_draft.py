# cSpell: disable=invalid-name
"""
Unit tests for trajectory estimator.
"""
# pylint: disable=no-name-in-module

import unittest

from gtsam import Cal3_S2, PinholeCameraCal3_S2, Point2, Point3, Pose3, Rot3
from gtsam.utils.test_case import GtsamTestCase
from localization.features import Features
from localization.observed_landmarks import ObservedLandmarks
from localization.trajectory_estimator import TrajectoryEstimator
from utilities.back_projection import back_projection


def calculate_vision_area(calibration, pose, depth, image_size):
    """Calculate the vision area of the current pose. 
        Return:
            vision_area - [[x,y,z]], back project the four image corners [0,0], [width-1,0],[width-1,height-1],[0,height-1]
    """
    vision_area = []
    point_1 = back_projection(calibration, Point2(0, 0), pose, depth)
    vision_area.append([point_1.x(), point_1.y(), point_1.z()])
    point_2 = back_projection(
        calibration, Point2(image_size[0]-1, 0), pose, depth)
    vision_area.append([point_2.x(), point_2.y(), point_2.z()])
    point_3 = back_projection(calibration, Point2(
        image_size[0]-1, image_size[1]-1), pose, depth)
    vision_area.append([point_3.x(), point_3.y(), point_3.z()])
    point_4 = back_projection(
        calibration, Point2(0, image_size[1]-1), pose, depth)
    vision_area.append([point_4.x(), point_4.y(), point_4.z()])

    return vision_area


class TestTrajectoryEstimator(GtsamTestCase):
    """Unit tests for trajectory estimator."""

    def setUp(self):
        # Camera to world rotation
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)  # pylint: disable=invalid-name
        initial_pose = Pose3(wRc, Point3(0, 0, 1.2))
        file_name = ""
        self.calibration = Cal3_S2(fx=1, fy=1, s=0, u0=320, v0=240)
        self.image_size = (640, 480)
        l2_thresh = 0.7
        distance_thresh = [5, 5]
        self.trajectory_estimator = TrajectoryEstimator(
            initial_pose, file_name, self.calibration, self.image_size, l2_thresh, distance_thresh)

    def test_landmark_projection(self):
        """Test landmark projection."""
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)  # pylint: disable=invalid-name
        pose = Pose3(wRc, Point3(0, 0, 1.2))
        depth = 10
        # Calculate the camera field of view area
        vision_area = calculate_vision_area(
            self.calibration, pose, depth, self.image_size)
        points = [vision_area[0], vision_area[1], vision_area[2],
                  vision_area[3], [1, 10, 10], [10, 10, 6000]]
        map_data = ObservedLandmarks(points, [[0, 0, 1], [0, 1, 0], [
                             0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]])

        self.trajectory_estimator.load_map(map_data)
        expected_observed_landmarks = ObservedLandmarks(
            points[:5],  [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]])
        actual_observed_landmarks = self.trajectory_estimator.landmark_projection(
            pose)
        if (expected_observed_landmarks == actual_observed_landmarks) is False:
            print("Expected_observed_landmarks: ", expected_observed_landmarks.landmarks,
                  expected_observed_landmarks.descriptors)
            print("Actual_observed_landmarks: ", actual_observed_landmarks.landmarks,
                  actual_observed_landmarks.descriptors)
        self.assertEqual(expected_observed_landmarks ==
                         actual_observed_landmarks, True)

    def test_landmark_association(self):
        """Test landmark association."""
        superpoint_features = Features([[10, 15], [300, 200], [11, 12], [14, 46]], [
                                       [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]])
        observed_landmarks = ObservedLandmarks([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], [
                                       [0, 0, 1], [1, 0, 0], [0, 2, 2], [0, 1, 0]])
        observed_landmarks.load_keypoints(
            [[8, 16], [14, 90], [114, 72], [300, 200]])
        expected_observations = [(Point2(10, 15), Point3(
            1, 1, 1)), (Point2(300, 200), Point3(4, 4, 4))]
        actual_observations = self.trajectory_estimator.landmark_association(
            superpoint_features, observed_landmarks)
        for i, observation in enumerate(expected_observations):
            self.gtsamAssertEquals(actual_observations[i][0], observation[0])
            self.gtsamAssertEquals(actual_observations[i][1], observation[1])

    def test_pose_estimate(self):
        """Test pose estimate."""
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)  # pylint: disable=invalid-name
        estimated_pose = Pose3(wRc, Point3(0, 0, 1.2))
        camera = PinholeCameraCal3_S2(estimated_pose, self.calibration)
        landmark_points = [[-5, 10, 5], [5, 10, 5], [5, 10, -5], [-5, 10, -5]]

        observations = []
        for landmark in landmark_points:
            # feature is gtsam.Point2 object
            landmark_point = Point3(landmark[0], landmark[1], landmark[2])
            feature_point = camera.project(landmark_point)
            observations.append((feature_point, landmark_point))

        current_pose = self.trajectory_estimator.pose_estimate(
            observations, estimated_pose)
        self.gtsamAssertEquals(current_pose, estimated_pose)


if __name__ == "__main__":
    unittest.main()
