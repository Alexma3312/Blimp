"""
Camera
"""
from gtsam import Cal3_S2
import numpy as np


class Camera():
    def __init__(self, fx, fy, u0, v0, image_size):
        self.distort_enable = False
        self.calibration = Cal3_S2(fx=fx, fy=fy, s=0, u0=u0, v0=v0)
        self.image_size = (image_size[1], image_size[0])

    def distorted_camera(self, raw_image_size, fx, fy, s, u0, v0, k1, k2, d1, d2, k3):
        """"""
        self.distort_enable = True
        self.raw_calibration = Cal3_S2(fx=fx, fy=fy, s=s, u0=u0, v0=v0)
        self.raw_distortion = np.array([k1, k2, d1, d2, k3])
        self.raw_image_size = (raw_image_size[1], raw_image_size[0])
        # TODO: Add undistort image size
        self.undistort_image_size = self.raw_image_size
        self.undistort_calibration = self.raw_calibration

    def undistort(self, image):
        """"""
        # TODO: finish undistort
        return image
