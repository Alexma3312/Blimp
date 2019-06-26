"""Run the mapping backend with the data from the Klaus auditorium."""
# cSpell: disable=invalid-name
# pylint: disable=no-name-in-module,wrong-import-order,no-member,assignment-from-no-return

import time

import numpy as np

import gtsam
from feature_matcher.mapping_back_end_dsf import MappingBackEnd
from gtsam import Cal3_S2  # pylint: disable=ungrouped-imports
from utilities.plotting import plot_with_result
from utilities.pose_estimate_generator import pose_estimate_generator


def run():
    """Execution."""
    # Input images(undistorted) calibration
    calibration = Cal3_S2(
        fx=232.0542, fy=252.8620, s=0, u0=325.3452, v0=240.2912)
    # Create pose estimates
    theta = 45
    delta_x = 1
    delta_y = -0.5
    delta_z = 1.2
    rows = 2
    cols = 2
    angles = 8

    prior1_delta = [0, -1, 1.2, 0]
    prior2_delta = [1, -1, 1.2, 0]
    pose_estimates = pose_estimate_generator(
        theta, delta_x, delta_y, delta_z, prior1_delta, prior2_delta, rows, cols, angles)

    # Create measurement noise for bundle adjustment
    sigma = 1.0
    measurement_noise = gtsam.noiseModel_Isotropic.Sigma(2, sigma)
    # Create pose prior noise
    rotation_sigma = np.radians(60)
    translation_sigma = 1
    pose_noise_sigmas = np.array([rotation_sigma, rotation_sigma, rotation_sigma,
                                  translation_sigma, translation_sigma, translation_sigma])
    pose_prior_noise = gtsam.noiseModel_Diagonal.Sigmas(pose_noise_sigmas)
    # Create MappingBackEnd instance
    data_directory = 'feature_matcher/library_data/library_4X8/undistort_images/features/'
    num_images = 34
    filter_bad_landmarks_enable = True
    min_obersvation_number = 3
    prob = 0.9
    threshold = 3
    backprojection_depth = 2
    back_end = MappingBackEnd(data_directory, num_images, calibration,
                              pose_estimates, measurement_noise, pose_prior_noise, filter_bad_landmarks_enable, min_obersvation_number, prob, threshold, backprojection_depth)
    # Bundle Adjustment
    tic_ba = time.time()
    sfm_result = back_end.bundle_adjustment()
    toc_ba = time.time()
    print('BA spents ', toc_ba-tic_ba, 's')
    # Plot Result
    plot_with_result(sfm_result, 5, 5, 5, 0.5)


if __name__ == "__main__":
    run()
