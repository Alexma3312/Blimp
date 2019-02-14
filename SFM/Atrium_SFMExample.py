import math
import unittest

import numpy as np

import gtsam
import gtsam.utils.visual_data_generator as generator
from gtsam import Point2, Point3, Pose3, symbol


def X(i):
    """Create key for pose i."""
    return gtsam.symbol(ord('x'), i)


def P(j):
    """Create key for landmark j."""
    return gtsam.symbol(ord('p'), j)

class AtriumSFMExample(object):

    def __init__(self):
        self.nrCameras = 3
        self.nrPoints = 5
        # Horizontal FOV = 128, Vertical FOV = 91, Diagonal FOV = 160
        # fov_in_degrees, w, h = 128, 160, 120
        fov_in_degrees, w, h = 128, 640, 480
        self.calibration = gtsam.Cal3_S2(fov_in_degrees, w, h)
        self.truth = []

    def generate_data(self):

        # data = [[Point2(88, 63), Point2(72, 64), Point2(61, 76), Point2(82, 99), Point2(92, 98)],
        #           [Point2(76, 74), Point2(60, 73), Point2(
        #               46, 86), Point2(59, 110), Point2(70, 110)],
        #           [Point2(86, 45), Point2(70, 42), Point2(56, 54), Point2(60, 77), Point2(70, 79)]]

        # data = [[Point2(352, 252), Point2(288, 256), Point2(244, 313), Point2(328, 396), Point2(368, 392),
        #             Point2(140, 188), Point2(328, 296), Point2(456, 164), Point2(520, 224), Point2(500, 140)],
        #           [Point2(304, 296), Point2(240, 292), Point2(184, 344), Point2(236, 440), Point2(280, 440),
        #           Point2(108, 208), Point2(272, 340), Point2(428, 220), Point2(480, 280), Point2(464, 184)],
        #           [Point2(344, 180), Point2(280, 168), Point2(224, 216), Point2(240, 308), Point2(280, 316),
        #           Point2(168, 92), Point2(308, 216), Point2(484, 124), Point2(516, 200), Point2(512, 92)]]

        data = [[Point2(352, 252), Point2(288, 256), Point2(244, 313), Point2(328, 396), Point2(368, 392)],
                [Point2(304, 296), Point2(240, 292), Point2(
                    184, 344), Point2(236, 440), Point2(280, 440)],
                [Point2(344, 180), Point2(280, 168), Point2(224, 216), Point2(240, 308), Point2(280, 316)]]
        return data

    def back_project(self, feature_point, calibration, depth):
        """back-project to 3D point at given depth, in camera coordinates."""
        pn = self.calibration.calibrate(feature_point)  # normalized
        return gtsam.Point3(depth, depth*pn.x(), 1.5-pn.y()*depth)

    def Atrium_SFMExample(self, data):
        
        # self.calibration = gtsam.Cal3_S2(50.0, 50.0, 0.0, 50.0, 50.0)

        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()

        # Add factors for all measurements
        measurementNoiseSigma = 1.0
        measurementNoise = gtsam.noiseModel_Isotropic.Sigma(
            2, measurementNoiseSigma)
        for i in range(len(data)):
            for j in range(len(data[i])):
                graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    data[i][j], measurementNoise,
                    X(i), P(j), self.calibration))

        # Create priors and initial estimate
        s = np.radians(0)
        poseNoiseSigmas = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
        posePriorNoise = gtsam.noiseModel_Diagonal.Sigmas(poseNoiseSigmas)
        angle = 0
        for i, y in enumerate([-1, 0, 1]):
            theta = np.radians(-y*angle)
            wRc = gtsam.Rot3(np.array([[0, math.cos(
                theta), -math.sin(theta)], [0, -math.sin(theta), -math.cos(theta)], [1, 0, 0]]).T)
            wTi = gtsam.Pose3(wRc, gtsam.Point3(0, (y+1)*2.5, 1.5))
            # graph.add(gtsam.PriorFactorPose3(X(i),
            #                                     wTi, posePriorNoise))
            initialEstimate.insert(X(i), wTi)

        graph.add(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(wRc, gtsam.Point3(0, 0, 1.5)), posePriorNoise))
        pointNoiseSigma = 0.1
        pointPriorNoise = gtsam.noiseModel_Isotropic.Sigma(3, pointNoiseSigma)
        graph.add(gtsam.PriorFactorPoint3(P(0),
                                Point3(10.0, 0.0, 0.0), pointPriorNoise))
        # graph.add(gtsam.PriorFactorPoint3(P(1),
        #                         Point3(10.0, 5.0, 0.0), pointPriorNoise))

        # Add initial estimates for Points.
        # for j in range(self.nrPoints):
        #     point_j = self.back_project(
        #         data[i][j], self.calibration, 10)
        #     initialEstimate.insert(P(j), point_j)

        initialEstimate.insert(P(0), Point3(10.0-0.25, 0.0+0.2, 0.0+0.15))
        initialEstimate.insert(P(1), Point3(10.0-0.25, 5.0+0.2, 0.0+0.15))
        initialEstimate.insert(P(2), Point3(10.0-0.25, 2.5+0.2, 2.5+0.15))
        initialEstimate.insert(P(3), Point3(15.0-0.25, 0.0+0.2, 5.0+0.15))
        initialEstimate.insert(P(4), Point3(15.0-0.25, 5.0+0.2, 5.0+0.15))

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        for i in range(5):
            optimizer.iterate()
        result = optimizer.values()

        # Marginalization
        marginals = gtsam.Marginals(graph, result)
        marginals.marginalCovariance(P(0))
        marginals.marginalCovariance(X(0))

        print(result)

        return result

    def Atrium_SFM(self, data):

        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()

        # Add factors for all measurements
        measurementNoiseSigma = 1.0
        measurementNoise = gtsam.noiseModel_Isotropic.Sigma(
            2, measurementNoiseSigma)
        for i in range(len(data)):
            for j in range(len(data[i])):
                graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    data[i][j], measurementNoise,
                    X(i), P(j), self.calibration))

        # Create priors and initial estimate        
        s = np.radians(30)
        # poseNoiseSigmas = np.array([s, s, s, 5, 5, 5])
        poseNoiseSigmas = np.array([0.3, 0.3, 0.3, 5, 5, 5])
        posePriorNoise = gtsam.noiseModel_Diagonal.Sigmas(poseNoiseSigmas)
        for i, y in enumerate([-1, 0, 1]):
            angle = 0
            theta = np.radians(-y*angle)
            wRc = gtsam.Rot3(np.array([[0, math.cos(
                theta), -math.sin(theta)], [0, -math.sin(theta), -math.cos(theta)], [1, 0, 0]]).T)
            wTi = gtsam.Pose3(wRc, gtsam.Point3(0, (y+1)*2.5, 1.5))
            # graph.add(gtsam.PriorFactorPose3(X(i),
            #                                  wTi, posePriorNoise))
            initialEstimate.insert(X(i), wTi)

        graph.add(gtsam.PriorFactorPose3(X(0),
                                             gtsam.Pose3(wRc, gtsam.Point3(0, 0, 1.5)), posePriorNoise))
        
        graph.add(gtsam.PriorFactorPose3(X(1),
                                             gtsam.Pose3(wRc, gtsam.Point3(0, 2.5, 1.5)), posePriorNoise))

        # Add initial estimates for Points.
        for j in range(self.nrPoints):
            point_j = self.back_project(
                data[i][j], self.calibration, 10)
            initialEstimate.insert(P(j), point_j)

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        for i in range(5):
            optimizer.iterate()
        result = optimizer.values()

        # Marginalization
        marginals = gtsam.Marginals(graph, result)
        marginals.marginalCovariance(P(0))
        marginals.marginalCovariance(X(0))

        print(result)

        ground_truth = self.truth

        return result, ground_truth


if __name__ == '__main__':
    AtriumSFMExample = AtriumSFMExample()
    data = AtriumSFMExample.generate_data()
    result, ground_truth = AtriumSFMExample.Atrium_SFM(data)
