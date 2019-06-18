"""Run the mapping backend with the data from the Klaus auditorium."""
# cSpell: disable=invalid-name
# pylint: disable=no-name-in-module,wrong-import-order,no-member,assignment-from-no-return

import time

import numpy as np

import gtsam
from feature_matcher.mapping_back_end_dsf import MappingBackEnd
from gtsam import Cal3_S2, Point3, Pose3, Rot3  # pylint: disable=ungrouped-imports
from utilities.plotting import plot_with_results


def run():
    """Execution."""
    # Input images(undistorted) calibration
    calibration = Cal3_S2(
        fx=232.0542, fy=252.8620, s=0, u0=325.3452, v0=240.2912)
    # Camera to world rotation
    wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)  # pylint: disable=invalid-name
    # Create pose estimates
    pose_estimates = [Pose3(wRc, Point3(1.58*i, 0, 1.2)) for i in range(6)]
    # Create measurement noise for bundle adjustment
    sigma = 1.0
    measurement_noise = gtsam.noiseModel_Isotropic.Sigma(2, sigma)
    # Create pose prior noise
    rotation_sigma = np.radians(60)
    translation_sigma = 1
    pose_noise_sigmas = np.array([rotation_sigma, rotation_sigma, rotation_sigma,
                                  translation_sigma, translation_sigma, translation_sigma])
    pose_prior_noise = gtsam.noiseModel_Diagonal.Sigmas(pose_noise_sigmas)

    #"""Use 4D agri matching"""
    data_directory = 'feature_matcher/Klaus_4d_agri_match_data/'
    num_images = 6
    min_obersvation_number = 6
    back_end = MappingBackEnd(data_directory, num_images, calibration,
                              pose_estimates, measurement_noise, pose_prior_noise, min_obersvation_number)
    # Bundle Adjustment
    tic_ba = time.time()
    sfm_result1 = back_end.bundle_adjustment()
    toc_ba = time.time()
    print('BA spents ', toc_ba-tic_ba, 's')

    #"""Use Superpoint matching"""
    # Create MappingBackEnd instance
    data_directory = 'feature_matcher/Klaus_Superpoint_match_data/'
    num_images = 6
    min_obersvation_number = 6
    back_end = MappingBackEnd(data_directory, num_images, calibration,
                              pose_estimates, measurement_noise, pose_prior_noise, min_obersvation_number)
    # Bundle Adjustment
    tic_ba = time.time()
    sfm_result2 = back_end.bundle_adjustment()
    toc_ba = time.time()
    print('BA spents ', toc_ba-tic_ba, 's')

    # Plot Result
    plot_with_results(sfm_result1, sfm_result2, 20, 20, 20)


if __name__ == "__main__":
    run()
