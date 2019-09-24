"""Undistortion utilities."""
import glob
# cSpell: disable=invalid-name
# pylint: disable=no-member
import os

import cv2


def undistort(basedir, img_extension, output_dir, output_prefix, calibration, distortion, output_image_shape=(640, 480), scaling_param=1):
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
        height, width, _ = img.shape
        new_camera_matrix = calibration

        # scaling parameter between 0 (when all the pixels in the undistorted image are valid)
        # and 1 (when all the source image pixels are retained in the undistorted image)
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
            calibration, distortion, (width, height), scaling_param, output_image_shape)
        print("calibration", calibration)
        print("new_camera_matrix", new_camera_matrix)

        # undistort
        mapx, mapy = cv2.initUndistortRectifyMap(
            calibration, distortion, None, new_camera_mtx, output_image_shape, 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        output_path = output_dir+output_prefix+'_%d' % img_idx+img_extension
        print(output_path)
        cv2.imwrite(output_path, dst)
    return True
