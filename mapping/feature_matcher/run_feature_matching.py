# cSpell: disable=invalid-name
"""Undistort collected images."""
# pylint: disable=no-name-in-module, wrong-import-order
from mapping.feature_matcher.feature_matcher import FeatureMatcher
from mapping.feature_matcher.myconfig import (basedir, calibration_matrix,
                                              feature_type, image_extension,
                                              matching_type, number_images,
                                              source_image_size, threshold,
                                              undistort_img_size)


def run():
    """Execution."""
    feature_matcher = FeatureMatcher(
        basedir, image_extension, source_image_size, number_images)
    feature_matcher.feature_matching(
        undistort_img_size, feature_type, matching_type, calibration_matrix, threshold)


if __name__ == "__main__":
    run()
