import unittest

import gtsam
import gtsam.utils.visual_data_generator as generator
from gtsam import Point2, Point3, Pose3, symbol
from SFM import Atrium_SFMExample as SFM
from SFM import SFMdata
import numpy as np
import math



class TestAtriumSFMEample(unittest.TestCase):

    def setUp(self):
        self.sfm = SFM.AtriumSFMExample()
        fov_in_degrees, w, h = 128, 640, 480
        self.calibration = gtsam.Cal3_S2(fov_in_degrees, w, h)
        # self.calibration = gtsam.Cal3_S2(50.0, 50.0, 0.0, 50.0, 50.0)
        self.nrCameras = 3
        self.nrPoints = 5
    
    def assertGtsamEquals(self, actual, expected, tol=1e-2):
        """Helper function that prints out actual and expected if not equal."""
        equal = actual.equals(expected, tol)
        if not equal:
            raise self.failureException(
                "Not equal:\n{}!={}".format(actual, expected))

    def test_back_project(self):
        actual_point = self.sfm.back_project(Point2(320,240), self.calibration, 10)
        expected_point = gtsam.Point3(10,0,1.5)
        self.assertGtsamEquals(actual_point, expected_point)


    def test_Atrium_SFMExample(self):

        actual_points = []
        actual_poses = []

        # Create the set of ground-truth landmarks
        points = SFMdata.createPoints()

        # Create the set of ground-truth poses
        poses = SFMdata.createPoses()
        
        # Create the five feature point data input for  Atrium_SFMExample()
        feature_data = [[Point2()]*self.nrPoints]*self.nrCameras

        for i, pose in enumerate(poses):
            for j, point in enumerate(points):
                camera = gtsam.PinholeCameraCal3_S2(pose, self.calibration)
                feature_data[i][j] = camera.project(point)

        result = self.sfm.Atrium_SFMExample(feature_data)

        for i in range(len(poses)):
            pose_i = result.atPose3(symbol(ord('x'), i))
            self.assertGtsamEquals(pose_i, poses[i], 0.5)

        for j in range(len(points)):
            point_j = result.atPoint3(symbol(ord('p'), j))
            self.assertGtsamEquals(point_j, points[j], 0.1)


if __name__ == "__main__":
    unittest.main()
