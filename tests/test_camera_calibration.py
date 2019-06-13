"""Unit Test camera model and calibration matrix by calculating re-projection errors."""
import math
import unittest
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gtsam
import numpy as np
from gtsam import Point2, Point3, Pose3, symbol, Rot3
import cv2

def X(i):
    """Create key for pose i."""
    return symbol(ord('x'), i)

def P(j):
    """Create key for landmark j."""
    return symbol(ord('p'), j)

def plot_sfm_result(result):

    # Declare an id for the figure
    fignum = 0

    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')
    plt.cla()

    # Plot points
    gtsam_plot.plot_3d_points(fignum, result, 'rx')

    # Plot cameras
    i = 0
    while result.exists(X(i)):
        pose_i = result.atPose3(X(i))
        gtsam_plot.plot_pose3(fignum, pose_i, 2)
        i += 1

    # draw
    axes.set_xlim3d(-30, 30)
    axes.set_ylim3d(-30, 30)
    axes.set_zlim3d(-30, 30)
    plt.legend()
    plt.show()

def back_projection(cal, key_point = Point2(), pose = Pose3(), depth = 10, undist = 0):
    # Normalize input key_point
    if undist == 0:
        pn = cal.calibrate(key_point,tol=1e-5)
    if undist == 1:
        pn = cal.calibrate(key_point)
    
    # Transfer normalized key_point into homogeneous coordinate and scale with depth
    ph = Point3(depth*pn.x(),depth*pn.y(),depth)

    # Transfer the point into the world coordinate
    return pose.transform_from(ph)

def bundle_adjustment(cal, kp1, kp2, undist = 0):
    """ Use GTSAM to solve Structure from Motion.
    """
    # Initialize factor Graph
    graph = gtsam.NonlinearFactorGraph()
    initialEstimate = gtsam.Values()

    # Add factors for all measurements
    measurementNoiseSigma = 1.0
    measurementNoise = gtsam.noiseModel_Isotropic.Sigma(
        2, measurementNoiseSigma)
    for i in range(6):
        if undist == 0:
            graph.add(gtsam.GenericProjectionFactorCal3DS2(
                    kp1[i], measurementNoise,
                    X(0), P(i), cal))
            graph.add(gtsam.GenericProjectionFactorCal3DS2(
                    kp2[i], measurementNoise,
                    X(1), P(i), cal))
        if undist == 1:
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    kp1[i], measurementNoise,
                    X(0), P(i), cal))
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    kp2[i], measurementNoise,
                    X(1), P(i), cal))

    # Create priors and initial estimate
    s = np.radians(60)
    poseNoiseSigmas = np.array([s, s, s, 1, 1, 0.1])
    # poseNoiseSigmas = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    posePriorNoise = gtsam.noiseModel_Diagonal.Sigmas(poseNoiseSigmas)

    # Create Rot3
    wRc = Rot3(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]).T)
    for i, y in enumerate([0,5*1.58]):
        # The approximate height measurement is 1.2
        wTi = Pose3(wRc, Point3(0, y, 1.2))
        initialEstimate.insert(X(i), wTi)
        graph.add(gtsam.PriorFactorPose3(X(i), Pose3(
        wRc, Point3(0, y, 1.2)), posePriorNoise))


    # Add initial estimates for Points.
    for j in range(6):
        # Generate initial estimates by back projecting feature points collected at the initial pose
        point_j = back_projection(cal,kp1[j],Pose3(wRc, Point3(0, 0, 1.2)),30, undist)
        initialEstimate.insert(P(j), point_j)

    # Optimization
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
    result = optimizer.optimize()

    # Why this is not working?
    tol_error= gtsam.reprojectionErrors(graph,result)
    print("tol_error: ",tol_error)

    # Marginalization
    marginals = gtsam.Marginals(graph, result)
    # print(result)
    plot_sfm_result(result)

    landmarks = []
    for i in range(6):
        landmarks.append(result.atPoint3(P(i)))
    poses = [result.atPose3(X(0)),result.atPose3(X(1))]

    error = graph.error(result)
    print("Graph error:", error)

    return landmarks, poses

class CameraCalibration(object):
    def __init__(self,choice):
        # ROS calibration result
        if choice == 1:
            cal_distort = gtsam.Cal3DS2(fx=347.820593, fy=329.096945,s=0,u0=295.717950, v0=222.964889,  k1=-0.284322, k2=0.055723, p1=0.006772, p2=0.005264)
        if choice == 2:
            cal_distort = gtsam.Cal3DS2(fx=333.4, fy=314.7,s=0,u0=303.6, v0=247.6,  k1=-0.282548, k2=0.054412, p1=-0.001882, p2=0.004796)
        if choice == 3:
            cal_distort = gtsam.Cal3DS2(fx=343.555173, fy=327.221818,s=0,u0=295.979699, v0=261.530851,  k1=-0.305247, k2=0.064438, p1=-0.007641, p2=0.006581)
        if choice == 4:
            cal_distort = gtsam.Cal3DS2(fx=384.768827, fy=365.994262,s=0,u0=293.450481, v0=269.045187,  k1=-0.350338, k2=0.086711, p1=-0.006112, p2=0.013082)
        # Matlab toolbox calibration result
        if choice == 5:
            cal_distort = gtsam.Cal3DS2(fx=331.0165, fy=310.4791,s=0,u0=332.7372, v0=248.5307,  k1=-0.3507, k2=0.1112, p1=8.6304e-04 , p2=-0.0018)
        # cam1_features = [Point2(293,307),Point2(292,348),Point2(292,364),Point2(328,307),Point2(327,347),Point2(326,362)]
        # cam2_features = [Point2(73,307),Point2(74,346),Point2(74,361),Point2(109,307),Point2(110,348),Point2(110,362)]        
        cam1_features = [Point2(348,293),Point2(348,332),Point2(348,348),Point2(388,291),Point2(388,332),Point2(388,348)]
        cam2_features = [Point2(213,311),Point2(214,350),Point2(213,365),Point2(249,313),Point2(250,352),Point2(251,368)]

        self.cal = cal_distort
        self.kp1 = cam1_features
        self.kp2 = cam2_features

        if choice == 5:
            # cal_undistort = gtsam.Cal3_S2(fx=331.6959, fy=310.4940,s=0,u0=334.6017, v0=250.2013)
            cal_undistort = gtsam.Cal3DS2(fx=331.6959, fy=310.4940,s=0,u0=334.6017, v0=250.2013,  k1=-0.0076, k2=0.0088, p1=-6.0889e-04 , p2=3.3046e-06)
            cam1_features_undistort = [Point2(302,289),Point2(303,324),Point2(303,338),Point2(335,339),Point2(334,324),Point2(333,288)]
            cam2_features_undistort = [Point2(249,222),Point2(249,257),Point2(249,270),Point2(278,272),Point2(278,257),Point2(277,222)]
        
        if choice == 1:
            cal_undistort = gtsam.Cal3_S2(fx=240.446564, fy=265.140778,s=0,u0=302.423680, v0=221.096494)
            # cal_undistort = gtsam.Cal3_S2(fx=232.0542, fy=252.8620,s=0,u0=325.3452, v0=240.2912)
            # cal_undistort = gtsam.Cal3DS2(fx=232.0542, fy=252.8620,s=0,u0=325.3452, v0=240.2912,  k1=-0.0076, k2=0.0088, p1=-6.0889e-04 , p2=3.3046e-06)

            # cam1_features_undistort = [Point2(302,289),Point2(303,324),Point2(303,338),Point2(335,339),Point2(334,324),Point2(333,288)]
            # cam2_features_undistort = [Point2(249,222),Point2(249,257),Point2(249,270),Point2(278,272),Point2(278,257),Point2(277,222)]
            cam1_features_undistort = [Point2(339,277),Point2(339,312),Point2(340,326),Point2(369,277),Point2(369,313),Point2(370,327)]
            cam2_features_undistort = [Point2(243,295),Point2(243,329),Point2(242,343),Point2(269,295),Point2(269,330),Point2(269,344)]

        self.cal_undist = cal_undistort
        self.kp1_undist = cam1_features_undistort
        self.kp2_undist = cam2_features_undistort
    
    def reprojection_error(self, landmarks, poses, undist = 0):
        """
        Calculate the reprojection error of the result from a SfM factor graph
        """
        kp1 = []
        kp2 = []
        for i in range(6):
            if undist == 0:
                kp1.append([self.kp1[i].x(),self.kp1[i].y()])
                kp2.append([self.kp2[i].x(),self.kp2[i].y()])
            if undist == 1:
                kp1.append([self.kp1_undist[i].x(),self.kp1_undist[i].y()])
                kp2.append([self.kp2_undist[i].x(),self.kp2_undist[i].y()])
        kp1 = np.expand_dims(np.array(kp1), axis=1)
        kp2 = np.expand_dims(np.array(kp2), axis=1)
        key_point = (kp1,kp2)


        total_error = 0

        if undist == 0:
            mtx = self.cal.K()
            dist = self.cal.k()
        if undist == 1:
            mtx = self.cal_undist.matrix()
            dist = np.array([0.0,0.0,0.0,0.0])
        
        for i,pose in enumerate(poses):
            # Get wRc, wtc
            R = pose.rotation().matrix()
            t = pose.translation().vector()
            # Invert R,t to obtain cRw, ctw
            R = R.T
            t = -1*np.dot(R,t)

            point_list = []
            for point in landmarks:
                point_list.append([point.x(),point.y(),point.z()]) 
            
            point_list = np.expand_dims(np.array(point_list), axis=1)
            imgpoint, _ = cv2.projectPoints(point_list, R, t, mtx, dist)
            error = cv2.norm(imgpoint,key_point[i], cv2.NORM_L2) /len(imgpoint)
            total_error += error
        mean_error = total_error/len(poses)
        return mean_error


class TestCameraCalibration(unittest.TestCase):
    
    def setUp(self):
        self.cam = CameraCalibration(1)

    def test_back_projection(self):
        cal = gtsam.Cal3DS2(fx=1, fy=1,s=0,u0=320, v0=240,  k1=0, k2=0, p1=0 , p2=0)
        point = Point2(320,240)
        pose = Pose3(Rot3(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]).T), Point3(0, 2, 1.2))
        actual_point = back_projection(cal,point, pose)
        actual_point.equals(Point3(10,2,1.2),tol = 1e-9)

    def test_reprojection(self):
        undist = 0
        landmarks, poses = bundle_adjustment(self.cam.cal, self.cam.kp1, self.cam.kp2, undist)
        actual_error = self.cam.reprojection_error(landmarks, poses, undist)
        print("Distort_reprojection_error:", actual_error)
        self.assertAlmostEqual(actual_error,0,delta = 0.05)

    def test_reprojection_undistort(self):
        undist = 1
        landmarks, poses = bundle_adjustment(self.cam.cal_undist, self.cam.kp1_undist, self.cam.kp2_undist, undist)
        actual_error = self.cam.reprojection_error(landmarks, poses, undist)
        print("Undistort_reprojection_error:", actual_error)
        self.assertAlmostEqual(actual_error,0,delta = 0.06)

        

if __name__ == "__main__":
    unittest.main()