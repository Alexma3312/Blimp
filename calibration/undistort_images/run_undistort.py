# cSpell: disable=invalid-name
"""Undistort collected images."""
# pylint: disable=no-name-in-module, wrong-import-order
import numpy as np

from calibration.undistort_images.undistort_images import undistort
from gtsam import Cal3_S2


def run():
    """Execution"""
    basedir = 'calibration/undistort_images/distort_images/'
    img_extension = '.jpg'
    output_dir = 'calibration/undistort_images/undistort_images/'
    output_prefix = 'undist_image'
    calibration = Cal3_S2(
        fx=347.820593, fy=329.096945, s=0, u0=295.717950, v0=222.964889).matrix()
    distortion = np.array(
        [-0.284322, 0.055723, 0.006772, 0.005264, 0.000000])
    undist_calibration = Cal3_S2(
        fx=232.0542, fy=252.8620, s=0, u0=325.3452, v0=240.2912).matrix()
    undistort(basedir, img_extension, output_dir,
              output_prefix, calibration, distortion, undist_calibration)


if __name__ == "__main__":
    run()
