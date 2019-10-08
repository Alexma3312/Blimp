"""Execution File"""
from shonan_averaging.shonan_helper import read_rotation_data, generate_rotation_edge, generate_g20_data_file
from shonan_averaging.configs.myconfig_relative_rotation_verify import *
from utilities.pose_estimate_generator import pose_estimate_generator_rectangle
from mapping.feature_matcher.feature_matcher import FeatureMatcher
from gtsam import Rot3, Point3, Pose3
from utilities.plotting import plot_poses


def run():
    # Create feature matches
    feature_matcher = FeatureMatcher(
        basedir, image_extension, source_image_size, number_images)
    feature_matcher.undistortion(
            distort_calibration_matrix, distortion_coefficients, resize_output)
    feature_matcher.feature_matching(
        undistort_img_size, feature_type, matching_type, calibration_matrix, threshold)
    # Create initial estimation.
    # pose_estimates = pose_estimate_generator_rectangle(
    #     theta, delta_x, delta_y, delta_z, prior1_delta, prior2_delta, rows, cols, angles)
    # rotation_dict = read_rotation_data(basedir)
    # edges = generate_rotation_edge(rotation_dict, pose_estimates)
    # generate_g20_data_file(basedir, pose_estimates, edges)


def verfiy_relative_rotation():
    """Plot relative rotation to verify OpenCV function."""
    rotation_dict = read_rotation_data(basedir)
    translation_list = [Point3(0, 0, 0), Point3(
        1, 0, 0), Point3(-1, 0, 0), Point3(0, 0, 1), Point3(0, 0, -1), ]
    wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)

    for i in range(number_images):
        for j in range(i+1, number_images):
            rotation = rotation_dict[(i,j)]
            pose_pair = [Pose3(wRc,translation_list[i]), Pose3()]

if __name__ == "__main__":
    # run()
    verfiy_relative_rotation()
