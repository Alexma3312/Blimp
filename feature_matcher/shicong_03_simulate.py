"""Run the mapping backend with the simulate data."""
# cSpell: disable=invalid-name
# pylint: disable=no-name-in-module,wrong-import-order,no-member,assignment-from-no-return

import time

import numpy as np

import gtsam
from feature_matcher.mapping_back_end import MappingBackEnd
from feature_matcher.plotting import plot_sfm_result
from gtsam import Cal3_S2, Point3, Pose3, Rot3


def run():
    """Execution."""
    # Input images(undistorted) calibration
    fov, w, h = 60, 1280, 720
    calibration = Cal3_S2(fov, w, h)
    # Camera to world rotation
    wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)  # pylint: disable=invalid-name
    # Create pose estimates
    pose_estimates = [Pose3(wRc, Point3(0, i, 2)) for i in range(3)]
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
    data_directory = 'feature_matcher/sim_match_data/'
    num_images = 3
    back_end = MappingBackEnd(data_directory, num_images, calibration,
                              pose_estimates, measurement_noise, pose_prior_noise)
    # Bundle Adjustment
    tic_ba = time.time()
    sfm_result, poses, points = back_end.bundle_adjustment()
    toc_ba = time.time()
    print('BA spents ', toc_ba-tic_ba, 's')
    # Plot Result
    plot_sfm_result(sfm_result, poses, points)


if __name__ == "__main__":
    run()
