# cSpell: disable=invalid-name
"""Matched Feature Visualization."""
# pylint: disable=no-name-in-module, wrong-import-order
import numpy as np

from utilities.visualize_matched_features import visualize_matched_features


def run():
    """Execution"""
    basedir = "feature_matcher/Klaus_Auditorium_data/"
    number_images = 98
    visualize_matched_features(basedir, number_images)


if __name__ == "__main__":
    run()
