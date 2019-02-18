"""
Structure from motion based on GTSAM
"""
import math
import unittest

import gtsam
import numpy as np
from gtsam import Point2, Point3, Pose3, symbol

from SFM import SFMdata

def X(i):
    """Create key for pose i."""
    return gtsam.symbol(ord('x'), i)


def P(j):
    """Create key for landmark j."""
    return gtsam.symbol(ord('p'), j)


class AtriumSFM(object):

    def __init__(self, nrCameras, nrPoints, fov_in_degrees, image_width, image_height):
        """
        Args:
            nrCameras -- Number of cameras
            nrPoints -- Number of landmarks 
            fov_in_degrees -- Horizontal FOV = 128, Vertical FOV = 91, Diagonal FOV = 160
            image width and height -- camera output image [640,480], Superpoint output image [160,120]
        """
        self.nrCameras = nrCameras
        self.nrPoints = nrPoints
        fov_in_degrees, w, h = fov_in_degrees, image_width, image_height
        self.calibration = gtsam.Cal3_S2(fov_in_degrees, w, h)
        self.truth = []

    def back_project(self, feature_point, calibration, depth):
        """back-project to 3D point at given depth, in camera coordinates."""
        pn = self.calibration.calibrate(feature_point)  # normalized
        return gtsam.Point3(depth, depth*(pn.x()), 1.5-(pn.y())*depth)

    def Atrium_SFM(self, data, rot_angle, y_distance):
        """
        Args:

        Returns:

        """
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()

        # Add factors for all measurements
        measurementNoiseSigma = 1.0
        measurementNoise = gtsam.noiseModel_Isotropic.Sigma(
            2, measurementNoiseSigma)
        for i in range(len(data.Z)):
            for k in range(len(data.Z[i])):
                print(k)
                print(data.Z[i][k])
                j = data.J[i][k]
                graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    data.Z[i][j], measurementNoise,
                    X(i), P(j), self.calibration))
                print(data.Z[i][j])

        # Create priors and initial estimate
        s = np.radians(30)
        poseNoiseSigmas = np.array([s, s, s, 5, 5, 5])
        posePriorNoise = gtsam.noiseModel_Diagonal.Sigmas(poseNoiseSigmas)

        for i, y in enumerate([-1, 0, 1]):
            theta = np.radians(-y*rot_angle)
            # wRc = gtsam.Rot3(np.array([[0, math.cos(
            #     theta), -math.sin(theta)], [0, -math.sin(theta), -math.cos(theta)], [1, 0, 0]]).T)
            wRc = gtsam.Rot3(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]).T)
            wTi = gtsam.Pose3(wRc, gtsam.Point3(0, (y+1)*y_distance, 1.5))
            # graph.add(gtsam.PriorFactorPose3(X(i),
            #                                     wTi, posePriorNoise))
            initialEstimate.insert(X(i), wTi)

        graph.add(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(
            wRc, gtsam.Point3(0, 0, 1.5)), posePriorNoise))
        graph.add(gtsam.PriorFactorPose3(X(2), gtsam.Pose3(
            wRc, gtsam.Point3(0, 5, 1.5)), posePriorNoise))

        # Add prior for the first point.
        # pointNoiseSigma = 5
        # pointPriorNoise = gtsam.noiseModel_Isotropic.Sigma(3, pointNoiseSigma)
        # graph.add(gtsam.PriorFactorPoint3(P(0),
        #                         Point3(10.0, 0.0, 0.0), pointPriorNoise))

        # Add initial estimates for Points.
        for j in range(self.nrPoints):
            point_j = self.back_project(
                data.Z[0][j], self.calibration, 10)
            initialEstimate.insert(P(j), point_j)

        # initialEstimate.insert(P(0), Point3(10.0-0.25, 0.0+0.2, 0.0+0.15))
        # initialEstimate.insert(P(1), Point3(10.0-0.25, 5.0+0.2, 0.0+0.15))
        # initialEstimate.insert(P(2), Point3(10.0-0.25, 2.5+0.2, 2.5+0.15))
        # initialEstimate.insert(P(3), Point3(10.0-0.25, 0.0+0.2, 5.0+0.15))
        # initialEstimate.insert(P(4), Point3(10.0-0.25, 5.0+0.2, 5.0+0.15))

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        # result = optimizer.optimize()

        for i in range(5):
            optimizer.iterate()
        result = optimizer.values()

        # Marginalization
        # marginals = gtsam.Marginals(graph, result)
        # marginals.marginalCovariance(X(0))
        # marginals.marginalCovariance(X(2))

        return result


if __name__ == '__main__':
    AtriumSFM = AtriumSFM(3, 5, 128, 640, 480)
    data = SFMdata.Data(3, 5)
    data.generate_data()
    result = AtriumSFM.Atrium_SFM(data, 0, 2.5)
    print(result)
