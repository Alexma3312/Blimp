import unittest
import gtsam
from gtsam import symbol
import numpy as np
import gtsam.utils.visual_data_generator as generator


class AtriumSFMExample(object):

    def __init__(self):
        self.nrCameras = 3
        self.nrPoints = 5
        # Horizontal FOV = 128, Vertical FOV = 91, Diagonal FOV = 160
        fov_in_degrees, w, h = 128, 160, 120
        self.calibration = gtsam.Cal3_S2(fov_in_degrees, w, h)
        self.truth = []

    def generate_data(self):

        data = generator.Data(self.calibration, 3, 5)

        data.Z = [[gtsam.Point2(88, 63), gtsam.Point2(72, 64), gtsam.Point2(61, 76), gtsam.Point2(82, 99), gtsam.Point2(92, 98)],
                  [gtsam.Point2(76, 74), gtsam.Point2(60, 73), gtsam.Point2(
                      46, 86), gtsam.Point2(59, 110), gtsam.Point2(70, 110)],
                  [gtsam.Point2(86, 45), gtsam.Point2(70, 42), gtsam.Point2(56, 54), gtsam.Point2(60, 77), gtsam.Point2(70, 79)]]
        data.J = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]

        return data

    def back_project(self, feature_point, calibration, depth):
        """back-project to 3D point at given depth, in camera coordinates."""
        pn = self.calibration.calibrate(feature_point)  # normalized
        return gtsam.Point3(depth, depth*pn.x(), 1.5-pn.y()*depth)

    def result_printout(self, result):

        return

    def Atrium_SFMExample(self, data):

        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()

        # Add factors for all measurements
        measurementNoiseSigma = 1.0
        measurementNoise = gtsam.noiseModel_Isotropic.Sigma(
            2, measurementNoiseSigma)
        for i in range(len(data.Z)):
            for k in range(len(data.Z[i])):
                j = data.J[i][k]
                graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    data.Z[i][k], measurementNoise,
                    symbol(ord('x'), i), symbol(ord('p'), j), data.K))

        # Create priors and initial estimate
        wRc = gtsam.Rot3(np.array([[0,1,0],[0,0,-1],[1,0,0]]).T)
        s = np.radians(30)
        poseNoiseSigmas = np.array([s, s, s, 5, 5, 5])
        posePriorNoise = gtsam.noiseModel_Diagonal.Sigmas(poseNoiseSigmas)
        for i, y in enumerate([0, 2.5, 5]):
            wTi = gtsam.Pose3(wRc, gtsam.Point3(0, y, 1.5))
            graph.add(gtsam.PriorFactorPose3(symbol(ord('x'), i),
                                            wTi, posePriorNoise))
            initialEstimate.insert(symbol(ord('x'), i), wTi)

        # Add initial estimates for Points.
        for j in range(self.nrPoints):
            point_j = AtriumSFMExample.back_project(data.Z[i][j], self.calibration, 10)
            initialEstimate.insert(symbol(ord('p'), j), point_j)

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        for i in range(5):
            optimizer.iterate()
        result = optimizer.values()
        print(result)
        # Marginalization
        marginals = gtsam.Marginals(graph, result)
        marginals.marginalCovariance(symbol(ord('p'), 0))
        marginals.marginalCovariance(symbol(ord('x'), 0))

        # # Print out result
        # for i in range(self.nrCameras):
        #     print(symbol(ord('x'), i), ":", result.atPose3(symbol(ord('x'), i)))
        #     # self.truth.points.append(result.atPose3(symbol(ord('x'), i)))

        # for j in range(self.nrPoints):
        #     print(symbol(ord('p'), 0), ":", result.atPoint3(symbol(ord('p'), 0)))
        #     # self.truth.cameras.append(gtsam.SimpleCamera.Lookat(result.atPoint3(symbol(ord('p'), 0)),
        #     #                                                     gtsam.Point3(),
        #     #                                                     gtsam.Point3(
        #     #                                                         0, 0, 1),
        #     #                                                     self.calibration))
        ground_truth = self.truth

        # Calculate odometry between cameras
        # for i in range(1, options.nrCameras):
        #     data.odometry[i] = truth.cameras[i - 1].pose().between(
        #         truth.cameras[i].pose())

        return result, ground_truth


if __name__ == '__main__':
    AtriumSFMExample = AtriumSFMExample()
    data = AtriumSFMExample.generate_data()
    result, ground_truth = AtriumSFMExample.Atrium_SFMExample(data)
