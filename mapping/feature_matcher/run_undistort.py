# cSpell: disable=invalid-name
"""Undistort collected images."""
# pylint: disable=no-name-in-module, wrong-import-order
from mapping.feature_matcher.feature_matcher import FeatureMatcher
from mapping.feature_matcher.myconfig import (basedir,
                                              distort_calibration_matrix,
                                              distortion_coefficients,
                                              image_extension, number_images,
                                              source_image_size, resize_output)


def run():
    """Execution"""
    feature_matcher = FeatureMatcher(
        basedir, image_extension, source_image_size, number_images)
    feature_matcher.undistortion(
        distort_calibration_matrix, distortion_coefficients, resize_output)


if __name__ == "__main__":
    run()
