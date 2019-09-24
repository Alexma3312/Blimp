# cSpell: disable=invalid-name
"""Undistort collected images."""
# pylint: disable=no-name-in-module, wrong-import-order
from mapping.feature_matcher.feature_matcher import FeatureMatcher
from mapping.feature_matcher.myconfig import (basedir,
                                              image_extension, number_images,
                                              source_image_size, feature_type, undistort_img_size)


def run():
    """Execution."""
    feature_matcher = FeatureMatcher(
        basedir, image_extension, source_image_size, number_images)
    feature_matcher.feature_extraction(undistort_img_size, feature_type)


if __name__ == "__main__":
    run()
