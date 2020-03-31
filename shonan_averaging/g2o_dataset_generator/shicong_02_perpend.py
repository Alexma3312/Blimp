"""Execution File"""
from shonan_averaging.shonan_helper import read_rotation_data,generate_rotation_edge,generate_g20_data_file
from shonan_averaging.configs.myconfig_perpend import *
from mapping.feature_matcher.feature_matcher import FeatureMatcher

def run():
    # Create feature matches
    feature_matcher = FeatureMatcher(
        basedir, image_extension, source_image_size, number_images)
    feature_matcher.feature_matching(
        undistort_img_size, feature_type, matching_type, calibration_matrix, threshold)
    # Create initial estimation.
    rotation_dict = read_rotation_data(basedir)
    edges = generate_rotation_edge(rotation_dict, pose_estimates)
    generate_g20_data_file(basedir, pose_estimates, edges)

if __name__ == "__main__":
    run()