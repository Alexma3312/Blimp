"""Test incremental visual SAM."""

import unittest

import math
import gtsam
from gtsam import Point2, Point3, Pose3
import numpy as np

import matplotlib.pyplot as plt
import cv2


class Features(object):
    """"A collection of 2D features."""

    def __init__(self, features):
        """ Create from a numpy array.
            Keyword arguments:
                features -- (n,2) numpy array of u,v coordinates
        """
        assert features.dtype == np.float, "expected float array"
        assert features.shape[0] >= 0, "need (n,2) array"
        assert features.shape[1] == 2, "need (n,2) array"
        self._features = [Point2(p[0], p[1]) for p in features]

    def len(self):
        """Number of features."""
        return len(self._features)

    def __iter__(self):
        """Return an iterator for points."""
        return self._features.__iter__()


class SFM(object):
    """Incremental SFM class."""

    def __init__(self, calibration):
        """Construct from calibration."""
        self._calibration = calibration

        # Define the camera observation noise model
        self.measurement_noise = gtsam.noiseModel_Isotropic.Sigma(
        2, 1.0)  # one pixel in u and v
        self.parameters = gtsam.ISAM2Params()
        self.parameters.setRelinearizeThreshold(0.01)
        self.parameters.setRelinearizeSkip(1)
        self.isam = gtsam.ISAM2(self.parameters)

        # Create a Factor Graph and Values to hold the new data
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()

    def back_project(self, feature_point, depth):
        """back-project to 3D point at given depth, in camera coordinates."""
        pn = self._calibration.calibrate(feature_point)  # normalized
        u, v = pn.x(), pn.y()
        return Point3(depth*u, depth*v, depth)

    def process_first_frame(self, features, depth=10.0):
        """ Process the first features, initializing point at a given depth.
            Keyword arguments:Features
                features -- something
                depth -- depth at which to initialize the features (default 10m) 
        """
        assert isinstance(features, Features)
        assert features.len() > 0
        self._points = [self.back_project(p, depth) for p in features]
        self._poses = np.array([[0,0]])

    def pose(self, i):
        """Return i^th pose."""
        return Pose3()

    def point(self, j):
        """Return j^th point."""
        return self._points[j]

    def X(self, i):
        """Create key for pose i."""
        return int(gtsam.symbol(ord('x'), i))

    def L(self, j):
        """Create key for landmark j."""
        return int(gtsam.symbol(ord('l'), j))

    def process_next_frame(self, features, returned_tracks):
        """ Process the next frame features, initializing feature points and current pose.
            Keyword arguments:Features
                features -- something
                returned_tracks -- track information
        """
        # TODO(Shicong): To complete the function
        points = []
        pose = [0,0]
        self._poses.append(pose)
        return pose,points

    def iSAM(self, pose, points):
        i = len(self._poses)
       # Add factors for each landmark observation
        for j, point in enumerate(points):
            camera = gtsam.PinholeCameraCal3_S2(pose, self._calibration)
            measurement = camera.project(point)
            self.graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
                measurement, self.measurement_noise, self.X(i), self.L(j), self._calibration))

        # Add an initial guess for the current pose
        # Intentionally initialize the variables off from the ground truth
        self.initial_estimate.insert(self.X(i), pose.compose(gtsam.Pose3(
            gtsam.Rot3.Rodrigues(-0.1, 0.2, 0.25), gtsam.Point3(0.05, -0.10, 0.20))))

        # If this is the first iteration, add a prior on the first pose to set the
        # coordinate frame and a prior on the first landmark to set the scale.
        # Also, as iSAM solves incrementally, we must wait until each is observed
        # at least twice before adding it to iSAM.
        if i == 0:
            # Add a prior on pose x0
            pose_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array(
                [0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))  # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
            self.graph.push_back(gtsam.PriorFactorPose3(self.X(0), self._poses[0], pose_noise))

            # Add a prior on landmark l0
            point_noise = gtsam.noiseModel_Isotropic.Sigma(3, 0.1)
            self.graph.push_back(gtsam.PriorFactorPoint3(
                self.L(0), points[0], point_noise))  # add directly to graph

            # Add initial guesses to all observed landmarks
            # Intentionally initialize the variables off from the ground truth
            for j, point in enumerate(points):
                self.initial_estimate.insert(self.L(j), gtsam.Point3(
                    point.x()-0.25, point.y()+0.20, point.z()+0.15))
        else:
            # Update iSAM with the new factors
            self.isam.update(self.graph, self.initial_estimate)
            # Each call to iSAM2 update(*) performs one iteration of the iterative nonlinear solver.
            # If accuracy is desired at the expense of time, update(*) can be called additional
            # times to perform multiple optimizer iterations every step.
            self.isam.update()
            current_estimate = self.isam.calculateEstimate()

    def iSAM_deconstruct(self):
        # Clear the factor graph and values for the next iteration
        self.graph.resize(0)
        self.initial_estimate.clear()

    



class TestVisualISAMExample(unittest.TestCase):

    def assertGtsamEquals(self, actual, expected, tol=1e-2):
        """Helper function that prints out actual and expected if not equal."""
        equal = actual.equals(expected, tol)
        if not equal:
            raise self.failureException(
                "Poses are not equal:\n{}!={}".format(actual, expected))

    def test_process_first_frame(self):
        # Horizontal FOV = 160
        fov_in_degrees, w, h = 160, 160, 120
        calibration = gtsam.Cal3_S2(fov_in_degrees, w, h)

        sfm = SFM(calibration)
        features = Features(np.array([[80, 60]], dtype=np.float))
        sfm.process_first_frame(features)
        self.assertGtsamEquals(sfm.pose(0), Pose3())

        # calculate expected point
        pn = calibration.calibrate(Point2(80.0, 60.0))  # normalized
        u, v = pn.x(), pn.y()
        print(u)
        print(v)
        depth = 10
        expected_point = Point3(depth*u, depth*v, depth)
        self.assertGtsamEquals(sfm.point(0), expected_point)


if __name__ == "__main__":
    unittest.main()
