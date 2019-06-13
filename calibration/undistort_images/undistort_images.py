"""Undistortion utilities."""
import glob
# cSpell: disable=invalid-name
# pylint: disable=no-member
import os

import cv2
import numpy as np


def undistort(basedir, img_extension, output_dir, output_prefix, calibration, distortion, undist_calibration, rectification=np.identity(3)):
    """A function to undistort the distorted images in a folder."""
    search = os.path.join(basedir, '*'+img_extension)
    img_paths = glob.glob(search)
    img_paths.sort()
    print("Number of Images: ", len(img_paths))
    maxlen = len(img_paths)
    if maxlen == 0:
        raise IOError(
            'No images were found (maybe wrong \'image extension\' parameter?)')

    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))

    for img_idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path, 1)
        dst = cv2.undistort(img, calibration, distortion)

        output_path = output_dir+output_prefix+'_%d' % img_idx+img_extension
        print(output_path)
        cv2.imwrite(output_path, dst)
    return True
