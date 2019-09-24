# cSpell: disable=invalid-name
"""Undistort collected images."""
# pylint: disable=no-name-in-module, wrong-import-order
from mapping.feature_matcher.feature_matcher import FeatureMatcher
from gtsam import Cal3_S2


def run():
    """Execution."""
    basedir = 'superpoint_descriptor/'
    image_extension = '*.jpg'
    image_size = (640, 480)
    number_images = 6
    # Input images(undistorted) calibration
    calibration = Cal3_S2(fx=232.0542, fy=252.8620,
                          s=0, u0=325.3452, v0=240.2912).matrix()
    feature_matcher = FeatureMatcher(
        basedir, image_extension, image_size, number_images)
    feature_matcher.feature_matching(
        image_size, 'Superpoint', 'FLANN', calibration)


if __name__ == "__main__":
    run()
