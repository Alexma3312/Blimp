"""Configuration"""
# pylint: disable=no-member, invalid-name
import numpy as np

from gtsam import Cal3_S2, Point3, Pose3, Rot3  # pylint: disable=no-name-in-module
import cv2
from utilities.pose_estimate_generator import pose_estimate_generator_rectangle_no_prior
from shonan_averaging.shonan_helper import read_shonan_result
from utilities.plotting import plot_poses

# Steps, True or False
run_undistortion = False
run_feature_extraction = False
run_feature_matching = False
run_bundle_adjustment = True
save_result = True
run_generate_g2o = True

basedir = "shonan_averaging/datasets/klaus_2x3x8_fisheye_new/"
image_extension = ".jpg"
source_image_size = (640, 480)

# Undistortion
# resize_output = True, output will be resized to the same shape as the input image
# resize_output = False, output image will shrink due to rectification of radian distort
resize_output = False
# Resize
# fx = 333.9190
# fy = 312.9898
# u0 = 330.9576
# v0 = 250.4098
fx = 199.2197
fy = 186.9630
u0 = 334.0790
v0 = 253.2759
calibration_matrix = Cal3_S2(fx=fx, fy=fy, s=0,
                             u0=u0, v0=v0)
undistort_img_size = (640, 480)

number_images = 48

# Feature Type can be:
#   - 'Superpoint'
#   - 'ORB'
feature_type = 'Superpoint'
feature_size = 256
# Superpoint parameter
nn_thresh = 0.7


# Matching Type can be:
#  - 'FLANN'
#  - 'Two Way NN'
matching_type = "FLANN"
output_prefix = "frame"

# RANSAC parameters
threshold = 1
prob = 0.999
method = cv2.RANSAC

# Create pose estimates
theta = 45
delta_x = [0, 3.7592, 5]
delta_y = [0, 1.75895, 1.75895*2, 1.75895*3]
delta_z = 0.9652
rows = 2
cols = 3
angles = 8

pose_estimates = pose_estimate_generator_rectangle_no_prior(
    theta, delta_x, delta_y, delta_z, rows, cols, angles)
# plot_poses(pose_estimates, 10, 10, 10, 1)
# wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)
# shonan_result = read_shonan_result(basedir, 'shonan_result.dat')
# shonan_result_normalize = [np.dot(shonan_result[0].matrix().transpose(
# ), shonan_result[i].matrix()) for i in range(number_images)]
# for rot in shonan_result_normalize:
#     print(np.dot(rot.transpose(), rot))
# pose_estimates = [Pose3(Rot3(np.dot(wRc.matrix(), shonan_result_normalize[i])), pose_estimates[i].translation())
#                   for i in range(number_images)]
# plot_poses(pose_estimates, 10, 10, 10, 1)

# Bundle Adjustment parameters
filter_bad_landmarks_enable = True
min_obersvation_number = 6
# There is result when backprojection_depth is 10. But the result is wrong.
backprojection_depth = 10
# Prior Indices
prior_indices = (0,8)

# Image Name: "raw_frame_row_col_angle"
source_directory = basedir+"source_images"

# Image Name: "frame_id"
undistort_folder = basedir+"undistort_images"

# 00000id.key
feature_folder = basedir+"features"

# match_i_j.dat
match_folder = basedir+"matches"