"""Configuration"""
# pylint: disable=no-member, invalid-name
import numpy as np
from utilities.pose_estimate_generator import pose_estimate_generator_rectangle
from gtsam import Cal3_S2, Pose3, Point3, Rot3 # pylint: disable=no-name-in-module
import cv2
from shonan_averaging.shonan_helper import read_shonan_result

# Steps
run_undistortion = False
run_feature_extraction = False
run_feature_matching = False
run_bundle_adjustment = True
save_result = False


basedir = "shonan_averaging/datasets/perpendicular_walls_manual_data/"
image_extension = ".jpg"
source_image_size = (640, 480)


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

number_images = 4

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
# Camera to world rotation
wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)
shonan_result = read_shonan_result(basedir,'shonan_result.dat')
shonan_result_normalize = [np.dot(shonan_result[i].matrix(),shonan_result[0].matrix().transpose()) for i in range(number_images)]
pose_estimates = [Pose3(Rot3(np.dot(shonan_result_normalize[i],wRc.matrix())), Point3(0.5*i, 0, 1.5))
                      for i in range(number_images)]

# pose_estimates = [Pose3(shonan_result[i], Point3(0.5*i, 0, 1.5))
#                       for i in range(number_images)]
# pose_estimates = [Pose3(Rot3(np.dot(shonan_result[i].matrix(),wRc.matrix())), Point3(0.5*i, 0, 1.5))
#                       for i in range(number_images)]

# Bundle Adjustment parameters
filter_bad_landmarks_enable = True
min_obersvation_number = 3
backprojection_depth = 2



# Image Name: "raw_frame_row_col_angle"
source_directory = basedir+"source_images"

# Image Name: "frame_id"
undistort_folder = basedir+"undistort_images"

# 00000id.key
feature_folder = basedir+"features"

# match_i_j.dat
match_folder = basedir+"matches"