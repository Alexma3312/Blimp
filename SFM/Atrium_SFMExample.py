import unittest
import gtsam
from gtsam import symbol
import numpy as np
import gtsam.utils.visual_data_generator as generator


class AtriumSFMExample(object):

    def __init__(self):
        self.nrCamera = 3
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
        u, v = pn.x(), pn.y()
        return gtsam.Point3(depth*u, depth*v, depth)

    def result_printout(self, result):

        return

    def Atrium_SFMExample(self, data):

        measurementNoiseSigma = 1.0
        pointNoiseSigma = 0.1
        poseNoiseSigmas = np.array([0.001, 0.001, 0.001, 0.1, 0.1, 0.1])

        graph = gtsam.NonlinearFactorGraph()

        # Add factors for all measurements
        measurementNoise = gtsam.noiseModel_Isotropic.Sigma(
            2, measurementNoiseSigma)
        for i in range(len(data.Z)):
            for k in range(len(data.Z[i])):
                j = data.J[i][k]
                graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    data.Z[i][k], measurementNoise,
                    symbol(ord('x'), i), symbol(ord('p'), j), data.K))

        posePriorNoise = gtsam.noiseModel_Diagonal.Sigmas(poseNoiseSigmas)
        graph.add(gtsam.PriorFactorPose3(symbol(ord('x'), 0),
                                         gtsam.Pose3(), posePriorNoise))
        pointPriorNoise = gtsam.noiseModel_Isotropic.Sigma(3, pointNoiseSigma)
        graph.add(gtsam.PriorFactorPoint3(symbol(ord('p'), 0),
                                          gtsam.Point3(0, 0, 0), pointPriorNoise))

        # Initial estimate
        initialEstimate = gtsam.Values()
        for i in range(self.nrCamera):
            pose_i = gtsam.Pose3(
                gtsam.Rot3.Rodrigues(-0.1, 0.2, 0.25), gtsam.Point3(0.05, -0.10, 0.20))
            initialEstimate.insert(symbol(ord('x'), i), pose_i)
        for j in range(self.nrPoints):
            point_j = AtriumSFMExample.back_project(data.Z[i][j], self.calibration, 10)
            initialEstimate.insert(symbol(ord('p'), j), point_j)


        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        for i in range(5):
            optimizer.iterate()
        result = optimizer.values()

        # Marginalization
        marginals = gtsam.Marginals(graph, result)
        marginals.marginalCovariance(symbol(ord('p'), 0))
        marginals.marginalCovariance(symbol(ord('x'), 0))

        # Print out result
        for i in range(self.nrCamera):
            print(symbol(ord('x'), i), ":", result.atPose3(symbol(ord('x'), i)))
            # self.truth.points.append(result.atPose3(symbol(ord('x'), i)))

        for j in range(self.nrPoints):
            print(symbol(ord('p'), 0), ":", result.atPoint3(symbol(ord('p'), 0)))
            # self.truth.cameras.append(gtsam.SimpleCamera.Lookat(result.atPoint3(symbol(ord('p'), 0)),
            #                                                     gtsam.Point3(),
            #                                                     gtsam.Point3(
            #                                                         0, 0, 1),
            #                                                     self.calibration))
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
