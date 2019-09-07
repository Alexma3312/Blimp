# cSpell: disable=invalid-name
"""Matched Feature Visualization.
Use create images with matched features on top of image pairs."""
# pylint: disable=no-name-in-module, wrong-import-order
from mapping.feature_matcher.myconfig import number_images, basedir

from utilities.visualize_matched_features import visualize_matched_features


def run():
    """Execution"""
    visualize_matched_features(basedir, number_images)


if __name__ == "__main__":
    run()
