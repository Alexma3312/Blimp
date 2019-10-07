"""Configuration"""
import cv2
# pylint: disable=no-member, invalid-name
import numpy as np

from gtsam import Cal3_S2, Point3, Pose3, Rot3
from shonan_averaging.shonan_helper import read_shonan_result
from utilities.plotting import plot_poses
from utilities.pose_estimate_generator import pose_estimate_generator_rectangle

run_undistortion = False
run_feature_extraction = False
run_feature_matching = False
run_bundle_adjustment = True
save_result = True

basedir = "shonan_averaging/datasets/klaus_4x3x1/"
image_extension = ".jpg"
source_image_size = (640, 480)


# Create pose estimates
theta = 45
delta_x = [0, 3.7592, 5]
delta_y = [0, 1.75895, 1.75895*2, 1.75895*3]
delta_z = 0.9652
rows = 4
cols = 3
angles = 1

prior1_delta = [0, 0, delta_z, 0]
prior2_delta = [3.7592, 1.75895, delta_z, 0]


# Undistortion
# resize_output = True, output will be resized to the same shape as the input image
# resize_output = False, output image will shrink due to rectification of radian distort
resize_output = False
distort_calibration_matrix = Cal3_S2(
    fx=347.820593, fy=329.096945, s=0, u0=295.717950, v0=222.964889).matrix()
distortion_coefficients = np.array(
    [-0.284322, 0.055723, 0.006772, 0.005264, 0.000000])
# calibration_matrix = Cal3_S2(fx=232.0542, fy=252.8620, s=0,
#                              u0=325.3452, v0=240.2912).matrix()
# Resize
calibration_matrix = Cal3_S2(fx=211.8927, fy=197.7030, s=0,
                             u0=281.1168, v0=179.2954)
undistort_img_size = (583, 377)

number_images = 14

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
rows = 4
cols = 3
angles = 1

prior1_delta = [0, 0, delta_z, 0]
prior2_delta = [3.7592, 1.75895, delta_z, 0]

# Create pose estimates
pose_estimates = pose_estimate_generator_rectangle(
    theta, delta_x, delta_y, delta_z, prior1_delta, prior2_delta, rows, cols, angles)
plot_poses(pose_estimates,10,10,10,1)
wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)
shonan_result = read_shonan_result(basedir, 'shonan_result_4x3x1.dat')
shonan_result_normalize = [np.dot(shonan_result[i].matrix(
), shonan_result[0].matrix().transpose()) for i in range(number_images)]
pose_estimates = [Pose3(Rot3(np.dot(shonan_result_normalize[i], wRc.matrix())), pose_estimates[i].translation())
                  for i in range(number_images)]
# plot_poses(pose_estimates,10,10,10,1)

# Bundle Adjustment parameters
filter_bad_landmarks_enable = True
min_obersvation_number = 6
backprojection_depth = 10
