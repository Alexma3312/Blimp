"""Run the mapping backend with the data from the Klaus auditorium."""
# cSpell: disable=invalid-name
# pylint: disable=no-name-in-module,wrong-import-order,no-member,assignment-from-no-return

import time

import numpy as np

import gtsam
from gtsam import Cal3_S2  # pylint: disable=ungrouped-imports
from mapping.bundle_adjustment.mapping_back_end_dsf import MappingBackEnd
from mapping.feature_matcher.feature_matcher import FeatureMatcher
from shonan_averaging.myconfig_4x3x5 import *
from utilities.plotting import plot_with_result

def run():
    """Execution."""
    # Undistortion
    if run_undistortion:
        feature_matcher = FeatureMatcher(
            basedir, image_extension, source_image_size, number_images)
        feature_matcher.undistortion(
            distort_calibration_matrix, distortion_coefficients, resize_output)
    # Feature Extraction
    feature_matcher = FeatureMatcher(
        basedir, image_extension, source_image_size, number_images)
    if run_feature_extraction:
        feature_matcher.feature_extraction(undistort_img_size, feature_type)
    # Feature Matching
    if run_feature_matching:
        feature_matcher.feature_matching(
            undistort_img_size, feature_type, matching_type, calibration_matrix)

    if run_bundle_adjustment:
        # Create measurement noise for bundle adjustment
        sigma = 1.0
        # measurement_noise = gtsam.noiseModel_Isotropic.Sigma(2, sigma)
        measurement_noise = gtsam.noiseModel_Robust(gtsam.noiseModel_mEstimator_Huber(
            1.345), gtsam.noiseModel_Isotropic.Sigma(2, sigma))

        # Create pose prior noise
        rotation_sigma = np.radians(60)
        translation_sigma = 1
        pose_noise_sigmas = np.array([rotation_sigma, rotation_sigma, rotation_sigma,
                                      translation_sigma, translation_sigma, translation_sigma])
        pose_prior_noise = gtsam.noiseModel_Diagonal.Sigmas(pose_noise_sigmas)
        # Create MappingBackEnd instance
        back_end = MappingBackEnd(basedir, number_images, calibration_matrix,
                                  pose_estimates, measurement_noise, pose_prior_noise, filter_bad_landmarks_enable, min_obersvation_number, prob, threshold, backprojection_depth)
        # Bundle Adjustment
        tic_ba = time.time()
        sfm_result = back_end.bundle_adjustment()
        toc_ba = time.time()
        print('BA spents ', toc_ba-tic_ba, 's')
        # Plot Result
        plot_with_result(sfm_result, 30, 30, 30, 0.5)

    # Save map data
    if save_result:
        back_end.save_map_to_file(sfm_result)
        back_end.save_poses_to_file(sfm_result)


if __name__ == "__main__":
    run()
