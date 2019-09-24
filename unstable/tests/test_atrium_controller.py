"""Test the Atrium Controller."""
import unittest
import cv2
import numpy as np
import gtsam
from gtsam import Point2, Point3, Pose3
import AtriumController

class TestAtriumControllerExample(unittest.TestCase):

    def setUp(self):
        # Create three black images
        size = 200, 200, 1
        self.image_1 = np.zeros(size, dtype=np.uint8)
        self.image_2 = np.zeros(size, dtype=np.uint8)
        self.image_3 = np.zeros(size, dtype=np.uint8)
        # Numpy array of point3
        # The trajectory will be
        # (0,0,0) -> (0,0,1) -> (0,3,1) -> (2,3,1) -> (2,3,0) -> (2,0,0) -> (0,0,0)
        self.old_estimate_trajectory_1 = np.array(
            [gtsam.Point3(0, 0, 0), gtsam.Point3(0, 0, 1), gtsam.Point3(0, 3, 1)])
        self.old_estimate_trajectory_2 = np.array(
            [gtsam.Point3(0, 0, 0), gtsam.Point3(0, 0, 1)])
        self.old_estimate_trajectory_3 = np.array([gtsam.Point3(0, 0, 0)])
        # Numpy array of point3
        # The map will include all the points in the world.
        # In this case, the map will include all the points of a cuboid (2*3*1)
        # (0,0,0), (0,0,1), (0,3,1), (2,3,1), (2,3,0), (2,0,0), (0,0,0)
        # (0,3,0), (2,0,1)
        self.atrium_map = np.array([gtsam.Point3(0, 0, 0), gtsam.Point3(0, 0, 1), gtsam.Point3(0, 3, 1),
                                    gtsam.Point3(2, 0, 1), gtsam.Point3(
                                        2, 3, 1), gtsam.Point3(2, 3, 0),
                                    gtsam.Point3(0, 3, 0), gtsam.Point3(2, 0, 0)])

        self.atrium_control = AtriumController.AtriumController()

    # This is a basic test function to test the script running
    def test_simple_atrium_controller(self):
        """
        This is a function to test the simple atrium controller.
        """
        expect_thrust_controller = 1
        expect_altitude_controller = 0
        expect_command = expect_thrust_controller, expect_altitude_controller

        actual_command = self.atrium_control.simple_atrium_controller()

        np.testing.assert_array_equal(expect_command, actual_command)

        return

    def test_atrium_controller(self):

        actual_command_1, actual_command_2, actual_command_3 = self.atrium_control.atrium_controller(
            self.image_1, self.image_2, self.image_3, self.old_estimate_trajectory_1, self.old_estimate_trajectory_2, self.old_estimate_trajectory_3, self.atrium_map)

        expect_command_1 = np.array([1, 0])
        expect_command_2 = np.array([2, 1])
        expect_command_3 = np.array([3, 2])

        np.testing.assert_array_equal(expect_command_1, actual_command_1)
        np.testing.assert_array_equal(expect_command_2, actual_command_2)
        np.testing.assert_array_equal(expect_command_3, actual_command_3)

        return


if __name__ == "__main__":
    unittest.main()
