"""Unit Test camera model and calibration matrix by calculating re-projection errors."""
# cSpell: disable=invalid-name
# pylint: disable=wrong-import-order,line-too-long
import unittest

import numpy as np

from calibration.reprojection_error_testing.camera_calibration import (
    CameraCalibration, back_projection, bundle_adjustment)
from gtsam import (Cal3_S2, Cal3DS2,  # pylint: disable=no-name-in-module
                   Point2, Point3, Pose3, Rot3)


def calibration_select(choice):
    """"Choose different calibration matrix to test."""
    #"""Distorted image variables setting"""
    # ROS calibration result
    if choice == 1:
        cal_distort = Cal3DS2(fx=347.820593, fy=329.096945, s=0, u0=295.717950,
                              v0=222.964889, k1=-0.284322, k2=0.055723, p1=0.006772, p2=0.005264)
    if choice == 2:
        cal_distort = Cal3DS2(fx=333.4, fy=314.7, s=0, u0=303.6,
                              v0=247.6, k1=-0.282548, k2=0.054412, p1=-0.001882, p2=0.004796)
    if choice == 3:
        cal_distort = Cal3DS2(fx=343.555173, fy=327.221818, s=0, u0=295.979699,
                              v0=261.530851, k1=-0.305247, k2=0.064438, p1=-0.007641, p2=0.006581)
    if choice == 4:
        cal_distort = Cal3DS2(fx=384.768827, fy=365.994262, s=0, u0=293.450481,
                              v0=269.045187, k1=-0.350338, k2=0.086711, p1=-0.006112, p2=0.013082)
    # Matlab toolbox calibration result
    if choice == 5:
        cal_distort = Cal3DS2(fx=331.0165, fy=310.4791, s=0, u0=332.7372,
                              v0=248.5307, k1=-0.3507, k2=0.1112, p1=8.6304e-04, p2=-0.0018)

    # Manually extracted features
    # cam1_features = [Point2(293,307),Point2(292,348),Point2(292,364),Point2(328,307),Point2(327,347),Point2(326,362)]
    # cam2_features = [Point2(73,307),Point2(74,346),Point2(74,361),Point2(109,307),Point2(110,348),Point2(110,362)]
    cam1_features = [Point2(348, 293), Point2(348, 332), Point2(
        348, 348), Point2(388, 291), Point2(388, 332), Point2(388, 348)]
    cam2_features = [Point2(213, 311), Point2(214, 350), Point2(
        213, 365), Point2(249, 313), Point2(250, 352), Point2(251, 368)]

    cal = cal_distort
    kp1 = cam1_features
    kp2 = cam2_features

    #"""Undistorted image variables setting"""
    if choice == 5:
        # cal_undistort = Cal3_S2(fx=331.6959, fy=310.4940,s=0,u0=334.6017, v0=250.2013)
        cal_undistort = Cal3DS2(fx=331.6959, fy=310.4940, s=0, u0=334.6017,
                                v0=250.2013,  k1=-0.0076, k2=0.0088, p1=-6.0889e-04, p2=3.3046e-06)
        cam1_features_undistort = [Point2(302, 289), Point2(303, 324), Point2(
            303, 338), Point2(335, 339), Point2(334, 324), Point2(333, 288)]
        cam2_features_undistort = [Point2(249, 222), Point2(249, 257), Point2(
            249, 270), Point2(278, 272), Point2(278, 257), Point2(277, 222)]

    if choice == 1:
        # cal_undistort = Cal3_S2(fx=240.446564, fy=265.140778,s=0,u0=302.423680, v0=221.096494)
        cal_undistort = Cal3_S2(
            fx=232.0542, fy=252.8620, s=0, u0=325.3452, v0=240.2912)
        # cal_undistort = Cal3DS2(fx=232.0542, fy=252.8620,s=0,u0=325.3452, v0=240.2912,  k1=-0.0076, k2=0.0088, p1=-6.0889e-04 , p2=3.3046e-06)

        # cam1_features_undistort = [Point2(302,289),Point2(303,324),Point2(303,338),Point2(335,339),Point2(334,324),Point2(333,288)]
        # cam2_features_undistort = [Point2(249,222),Point2(249,257),Point2(249,270),Point2(278,272),Point2(278,257),Point2(277,222)]
        cam1_features_undistort = [Point2(339, 277), Point2(339, 312), Point2(
            340, 326), Point2(369, 277), Point2(369, 313), Point2(370, 327)]
        cam2_features_undistort = [Point2(243, 295), Point2(243, 329), Point2(
            242, 343), Point2(269, 295), Point2(269, 330), Point2(269, 344)]

    cal_undist = cal_undistort
    kp1_undist = cam1_features_undistort
    kp2_undist = cam2_features_undistort

    return [cal, kp1, kp2], [cal_undist, kp1_undist, kp2_undist]


class TestCameraCalibration(unittest.TestCase):
    """test camera calibration"""

    def setUp(self):
        dist_variables, undist_variables = calibration_select(1)
        #"""Distorted image variables setting"""
        cal_distort = dist_variables[0]
        cam1_features = dist_variables[1]
        cam2_features = dist_variables[2]

        #"""Undistorted image variables setting"""
        cal_undistort = undist_variables[0]
        cam1_features_undistort = undist_variables[1]
        cam2_features_undistort = undist_variables[2]
        self.calib = CameraCalibration(cal_distort, cam1_features, cam2_features,
                                       cal_undistort, cam1_features_undistort, cam2_features_undistort)

    def test_back_projection(self):
        """test back projection"""
        cal = Cal3DS2(fx=1, fy=1, s=0, u0=320,
                      v0=240, k1=0, k2=0, p1=0, p2=0)
        point = Point2(320, 240)
        pose = Pose3(
            Rot3(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]).T), Point3(0, 2, 1.2))
        actual_point = back_projection(cal, point, pose)
        actual_point.equals(Point3(10, 2, 1.2), tol=1e-9)

    def test_reprojection(self):
        """test reprojection for for distort images"""
        undist = 0
        landmarks, poses = bundle_adjustment(
            self.calib.cal, self.calib.kp1, self.calib.kp2, undist)
        actual_error = self.calib.reprojection_error(landmarks, poses, undist)
        print("Distort_reprojection_error:", actual_error)
        self.assertAlmostEqual(actual_error, 0, delta=0.05)

    def test_reprojection_undistort(self):
        """test reprojection for undistort images"""
        undist = 1
        landmarks, poses = bundle_adjustment(
            self.calib.cal_undist, self.calib.kp1_undist, self.calib.kp2_undist, undist)
        actual_error = self.calib.reprojection_error(landmarks, poses, undist)
        print("Undistort_reprojection_error:", actual_error)
        self.assertAlmostEqual(actual_error, 0, delta=0.06)


if __name__ == "__main__":
    unittest.main()
