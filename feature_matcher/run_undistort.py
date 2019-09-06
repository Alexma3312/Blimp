# cSpell: disable=invalid-name
"""Undistort collected images."""
# pylint: disable=no-name-in-module, wrong-import-order
# More information: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
import numpy as np

from calibration.undistort_images.undistort_images_reserve_source_points import undistort as undistort_no_interpolate
from gtsam import Cal3_S2

def run():
    """Execution"""
    basedir = "feature_matcher/Klaus_Auditorium_data/source_images/"
    img_extension = ".jpg"
    output_dir = "feature_matcher/Klaus_Auditorium_data/undistort_images/"
    output_prefix = "frame"
    calibration = Cal3_S2(
        fx=347.820593, fy=329.096945, s=0, u0=295.717950, v0=222.964889).matrix()
    distortion = np.array(
        [-0.284322, 0.055723, 0.006772, 0.005264, 0.000000])
    # undist_calibration = Cal3_S2(
    #     fx=232.0542, fy=252.8620, s=0, u0=325.3452, v0=240.2912).matrix()
    undistort_no_interpolate(basedir, img_extension, output_dir, output_prefix,
              calibration, distortion)

if __name__ == "__main__":
    run()