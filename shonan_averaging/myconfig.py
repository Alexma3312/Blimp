"""Configuration"""
# pylint: disable=no-member, invalid-name
import numpy as np

from gtsam import Cal3_S2
import cv2

basedir = "shonan_averaging/datasets/klaus_4x3x1/"
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


# Folder Name: "/mapping/datasets/" + ""+"/"
# Image Name: "raw_frame_row_col_angle"
source_directory = basedir+"source_images"

#
undistort_folder = ""

#
feature_folder = ""

#
match_folder = ""
