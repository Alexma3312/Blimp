"""
Camera
"""
from gtsam import Cal3_S2
import numpy as np


class Camera():
    def __init__(self, calibration, distortion, image_size, distort_enable=True):
        """
        Parameters:
            image size: (width,height) to follow cv
        Member Variables:
            distort_enable: if distort enable is False, the camera calibration matrix will be a simple pinhole camera calibration matrix
            calibration: Cal3S2
            image size:(height, width) to follow numpy
        """
        self.distort_enable = distort_enable
        self.calibration = calibration
        self.distortion = distortion
        self.image_size = (image_size[1], image_size[0])

    # def distorted_camera(self, raw_image_size, fx, fy, s, u0, v0, k1, k2, d1, d2, k3):
    #     """"""
    #     self.distort_enable = True
    #     self.raw_calibration = Cal3_S2(fx=fx, fy=fy, s=s, u0=u0, v0=v0)
    #     self.raw_distortion = np.array([k1, k2, d1, d2, k3])
    #     self.raw_image_size = (raw_image_size[1], raw_image_size[0])
    #     # TODO: Add undistort image size
    #     self.undistort_image_size = self.raw_image_size
    #     self.undistort_calibration = self.raw_calibration

    # def undistort(self, image):
    #     """"""
    #     # TODO: finish undistort
    #     return image
