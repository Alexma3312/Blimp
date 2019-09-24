import math
import sys

import cv2
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
from gtsam import Point2, Point3, Pose3, symbol
from mpl_toolkits.mplot3d import Axes3D

from atrium_control.trajectory_estimator import *
from datasets.mapping import Data, Mapping
from sfm import sim3

sys.path.append('../')


class FrontEnd(object):
    def __init__(self, image_directory_path='datasets/4.5/localization/', img_extension='*.jpg', image_size=(160, 120), down_sample=4):
        self.basedir = image_directory_path
        self.img_extension = img_extension
        self.img_size = image_size
        self.scale = down_sample

        self.image_list = []

    def read_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs:
            impath - Path to input image.
            img_size - (W, H) tuple specifying resize size.
        Returns:
            grayim - float32 numpy array sized H x W with values in range [0, 1].
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        # Image is resized via opencv.
        interp = cv2.INTER_AREA
        grayim = cv2.resize(
            grayim, (self.img_size[1], self.img_size[0]), interpolation=interp)
        grayim = (grayim.astype('float32') / 255.)
        return grayim

    def get_image_paths(self):
        """Get all image paths within the directory."""
        print('==> Processing Image Directory Input.')
        search = os.path.join(self.basedir, self.img_extension)
        self.img_paths = glob.glob(search)
        self.img_paths.sort()
        maxlen = len(self.img_paths)
        if maxlen == 0:
            raise IOError(
                'No images were found (maybe wrong \'image extension\' parameter?)')

    def read_images(self):
        """ Read images and store image in an image list
        Parameters:
            downsample_width - image width downsample factor
            downsample_height - image height downsample factor
            width - input image width
            height - input image height
        Returns:
            image_list - a list of images
        """

        for img_path in self.img_paths:
            image = self.read_image(img_path)
            self.image_list.append(image)


def sim3_transform_map(result, nrCameras, nrPoints):
    # Similarity transform
    s_poses = []
    s_points = []
    _sim3 = sim3.Similarity3()

    for i in range(nrCameras):
        pose_i = result.atPose3(symbol(ord('x'), i))
        s_poses.append(pose_i)

    for j in range(nrPoints):
        point_j = result.atPoint3(symbol(ord('p'), j))
        s_points.append(point_j)

    theta = np.radians(30)
    wRc = gtsam.Rot3(np.array([[-math.sin(theta), 0, math.cos(theta)],
                               [-math.cos(theta), 0, -math.sin(theta)],
                               [0, 1, 0]]))
    d_pose1 = gtsam.Pose3(
        wRc, gtsam.Point3(-math.sin(theta)*1.58, -math.cos(theta)*1.58, 1.2))
    d_pose2 = gtsam.Pose3(
        wRc, gtsam.Point3(-math.sin(theta)*1.58*3, -math.cos(theta)*1.58*3, 1.2))
    pose_pairs = [[s_poses[2], d_pose2], [s_poses[0], d_pose1]]

    _sim3.align_pose(pose_pairs)
    print('R', _sim3._R)
    print('t', _sim3._t)
    print('s', _sim3._s)

    s_map = (s_poses, s_points)
    actual_poses, actual_points = _sim3.map_transform(s_map)
    d_map = _sim3.map_transform(s_map)

    return _sim3, d_map


def sim3_transform_multi_poses(result, nrCameras, nrPoints):
    # Similarity transform
    s_poses = []
    s_points = []
    _sim3 = sim3.Similarity3()

    for i in range(nrCameras):
        pose_i = result.atPose3(symbol(ord('x'), i))
        s_poses.append(pose_i)

    for j in range(nrPoints):
        point_j = result.atPoint3(symbol(ord('p'), j))
        s_points.append(point_j)

    theta = np.radians(30)
    wRc = gtsam.Rot3(np.array([[-math.sin(theta), 0, math.cos(theta)],
                               [-math.cos(theta), 0, -math.sin(theta)],
                               [0, 1, 0]]))
    d_pose0 = gtsam.Pose3(
        wRc, gtsam.Point3(-math.sin(theta)*1.58, -math.cos(theta)*1.58, 1.2))
    d_pose1 = gtsam.Pose3(
        wRc, gtsam.Point3(-math.sin(theta)*1.58*2, -math.cos(theta)*1.58*2, 1.2))
    d_pose2 = gtsam.Pose3(
        wRc, gtsam.Point3(-math.sin(theta)*1.58*3, -math.cos(theta)*1.58*3, 1.2))
    d_pose3 = gtsam.Pose3(
        wRc, gtsam.Point3(-math.sin(theta)*1.58*4, -math.cos(theta)*1.58*4, 1.2))
    d_pose4 = gtsam.Pose3(
        wRc, gtsam.Point3(-math.sin(theta)*1.58*5, -math.cos(theta)*1.58*5, 1.2))
    pose_pairs = [[s_poses[0], d_pose0], [s_poses[1], d_pose1], [s_poses[2], d_pose2], [
        s_poses[3], d_pose3], [s_poses[4], d_pose4]]

    _sim3.align_multi_poses(pose_pairs)
    print('R', _sim3._R)
    print('t', _sim3._t)
    print('s', _sim3._s)

    s_map = (s_poses, s_points)
    actual_poses, actual_points = _sim3.map_transform(s_map)
    d_map = _sim3.map_transform(s_map)

    return _sim3, d_map


def plot_localization(fignum, result):

    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')
    plt.cla()

    # Plot points
    # Can't use data because current frame might not see all points
    # marginals = Marginals(isam.getFactorsUnsafe(), isam.calculateEstimate())
    # gtsam.plot_3d_points(result, [], marginals)
    for point in result[1]:
        gtsam_plot.plot_point3(fignum, point, 'rx')

    # Plot cameras
    for pose in result[0]:
        gtsam_plot.plot_pose3(fignum, pose, 1)

    # draw
    axes.set_xlim3d(-10, 10)
    axes.set_ylim3d(-10, 10)
    axes.set_zlim3d(-10, 10)
    plt.pause(1)


def create_map(fignum):
    calibration = gtsam.Cal3_S2(fx=333.4, fy=314.7,s=0,u0=303.6, v0=247.6)
    distortion = np.array([-0.282548, 0.054412, -0.001882, 0.004796, 0.000000])
    rectification = np.identity(3)
    projection = np.array([[226.994629 ,0.000000, 311.982613, 0.000000],[0.000000, 245.874146, 250.410089, 0.000000],[0.000000, 0.000000, 1.000000, 0.000000]])
    # Create matched feature data and desc data
    data = Data(5,10,calibration.matrix(),distortion,projection)
    data.create_feature_data(0)
    data.create_descriptor_data()
    data.undistort_feature_points()

    # Mapping
    mapping = Mapping(5, 10, calibration, 640, 480)
    sfm_result = mapping.atrium_sfm(data,30,1.58)

    atrium_map = mapping.create_atrium_map(sfm_result, data.desc)

    return sfm_result, atrium_map


if __name__ == "__main__":
    plt.ion()
    fignum = 0

    # Create Map
    sfm_result, atrium_map = create_map(fignum)

    # sim3, d_map = sim3_transform_map(sfm_result, 5, 10)
    # plot_localization(fignum, d_map)

    # Get images
    fe = FrontEnd()
    fe.get_image_paths()
    fe.read_images()

    estimator = TrajectoryEstimator(
        atrium_map, 4, 4, 1, 100)
    estimator.trajectory = atrium_map.trajectory
    print("traj:",estimator.trajectory)

    counter = 0

    for i,image in enumerate(fe.image_list):
    # for i in range(1,2):
    #     image = fe.image_list[1]
        assert counter != 3, "Localization Failed"
        superpoint_features = estimator.superpoint_generator(image)
        # print(superpoint_features.key_points)
        pre_pose = estimator.trajectory[len(estimator.atrium_map.trajectory)-1]

        projected_features, map_indices = estimator.landmarks_projection(
            pre_pose)
        # print(projected_features.key_points)
        # print(map_indices)
        features, visible_map = estimator.data_association(
            superpoint_features, projected_features, map_indices)
        # if features.get_length() < 5:
        #     counter += 1
        #     print("Not enough correspondences.")
        #     continue
        # elif counter > 0:
        #     counter = 0

        # print(features.key_points)
        cur_pose = estimator.trajectory_estimator(features, visible_map)
        # gtsam_plot.plot_pose3(fignum, sim3.pose(cur_pose), 5)
        gtsam_plot.plot_pose3(fignum, cur_pose, 5)
        print("cur_pose",i,":",cur_pose)
    # # print(estimator.atrium_map.trajectory)

    plt.ioff()
    plt.show()
