import numpy as np

import gtsam
from evaluation.read_colmap_pose import read_colmap_pose_result
from evaluation.read_pose_estimation_result import read_pose_estimation_result
from gtsam import Pose3, Rot3
from sfm.sim3 import Similarity3
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from matplotlib import patches


def plot_pose3_on_axes(axes, pose, r, g, b, axis_length=0.1):
    """Plot a 3D pose on given axis 'axes' with given 'axis_length'."""
    # get rotation and translation (center)
    gRp = pose.rotation().matrix()  # rotation from pose to global
    t = pose.translation()
    origin = np.array([t.x(), t.y(), t.z()])

    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], r)

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], g)

    z_axis = origin + gRp[:, 2] * axis_length
    line = np.append(origin[np.newaxis], z_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], b)


def plot(align_estimate_poses, ground_truth_poses):
    fig = plt.figure(1)
    axes = fig.gca(projection='3d')
    length = len(align_estimate_poses)
    print(length)
    plot_pose3_on_axes(axes, align_estimate_poses[0], 'r-', 'r-', 'r-', axis_length=0.01)
    plot_pose3_on_axes(axes, ground_truth_poses[0], 'r-', 'g-', 'b-', axis_length=0.5)
    for i in range(1, length):
        align_estimate_pose = align_estimate_poses[i]
        ground_truth_pose = ground_truth_poses[i]
        plot_pose3_on_axes(axes, align_estimate_pose, 'r-', 'r-', 'r-', axis_length=0.01)
        plot_pose3_on_axes(axes, ground_truth_pose, 'g-', 'g-', 'g-', axis_length=0.01)
    # fig.set_facecolor('black')
    # axes.set_facecolor('black') 
    # axes.grid(False)
    # axes.w_xaxis.pane.fill = False
    # axes.w_yaxis.pane.fill = False
    # axes.w_zaxis.pane.fill = False
    # axes.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    # axes.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    # axes.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    plt.show()


def get_distance_angle_error(poses_dat_file, ground_truth_image_text_file, colmap_image_text_file, number_poses, sample_rate, map_number_poses):
    # there are 534 images, 0~136 are mapping images, 137~523 are ground truth poses
    colmap_poses, colmap_bad_indices = read_colmap_pose_result(ground_truth_image_text_file, map_number_poses+number_poses, map_number_poses, map_number_poses+number_poses-1)

    # Read pose estimation result, (5 is the sample rate)
    estimated_poses, estimate_bad_indices = read_pose_estimation_result(poses_dat_file, number_poses, sample_rate)
    # sim3 transform
    # Read colmap map poses
    groundtruth_map_poses, groundtruth_map_bad_indices = read_colmap_pose_result(ground_truth_image_text_file, map_number_poses, 0, map_number_poses-1)
    map_poses, map_bad_indices = read_colmap_pose_result(colmap_image_text_file, map_number_poses, 0, map_number_poses-1)

    # Generate pose pairs
    bad_indices = groundtruth_map_bad_indices | map_bad_indices
    pose_pairs = []
    for i in range(map_number_poses):
        if i not in bad_indices:
            pose_pairs.append([map_poses[i],groundtruth_map_poses[i]])
    # Get sim3 transform parameter
    sim3 = Similarity3()
    sim3.sim3_pose(pose_pairs)

    # Transform estimated poses to groundtruth poses
    bad_indices = colmap_bad_indices | estimate_bad_indices
    distances = []
    angles = []
    align_estimate_poses = []
    ground_truth_poses = []
    for i in range(number_poses):
        if i not in bad_indices:
            align_estimated_pose = sim3.pose(estimated_poses[i])
            align_estimate_poses.append(align_estimated_pose)
            ground_truth_poses.append(colmap_poses[i])

    plot(align_estimate_poses, ground_truth_poses)
    return distances, angles


poses_dat_file = "/home/sma96/Blimps/evaluation/face_inside_sift_superpoint_compare/sift_result/poses.dat"
ground_truth_image_text_file = "/home/sma96/Blimps/evaluation/face_inside_sift_superpoint_compare/ground_truth/images.txt"
colmap_image_text_file = "/home/sma96/Blimps/evaluation/face_inside_sift_superpoint_compare/sift_map/images.txt"
number_poses = 387
sample_rate = 5
map_number_poses = 137
distances_sift, angles_sift = get_distance_angle_error(
    poses_dat_file, ground_truth_image_text_file, colmap_image_text_file, number_poses, sample_rate, map_number_poses)

poses_dat_file = "/home/sma96/Blimps/evaluation/face_inside_sift_superpoint_compare/superpoint_result/poses.dat"
ground_truth_image_text_file = "/home/sma96/Blimps/evaluation/face_inside_sift_superpoint_compare/ground_truth/images.txt"
colmap_image_text_file = "/home/sma96/Blimps/evaluation/face_inside_sift_superpoint_compare/superpoint_map/images.txt"
number_poses = 387
sample_rate = 5
map_number_poses = 137
distances_superpoint, angles_superpoint = get_distance_angle_error(
    poses_dat_file, ground_truth_image_text_file, colmap_image_text_file, number_poses, sample_rate, map_number_poses)