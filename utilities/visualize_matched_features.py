# pylint: disable=no-member, line-too-long
# cspell: disable
"""
This script plot the matched features within a pair of images by displaying the image pair and all matches.
"""
import os

import cv2
import numpy as np

from mapping.bundle_adjustment.parser import get_matches, load_features

def read_features(basedir, image_index):
    """ Load features from .key files
        features - keypoints:A N length list of gtsam.Point2(x,y). Descriptors: A Nx256 list of descriptors.
    """
    feat_file = os.path.join(
        basedir, "{0:07}.key".format(image_index))
    keypoints, descriptors = load_features(feat_file)
    return keypoints, descriptors


def read_matches(basedir, frame_1, frame_2):
    """ Load matches from .dat files
        matches - a list of [image 1 index, image 1 keypoint index, image 2 index, image 2 keypoint index]
    """
    matches_file = os.path.join(
        basedir, "match_{0}_{1}.dat".format(frame_1, frame_2))
    if os.path.isfile(matches_file) is False:
        return []
    _, matches = get_matches(matches_file)
    return matches


def save_match_images(basedir, idx1, idx2, matches, keypoints_1, keypoints_2):
    """Create an image to display matches between an image pair."""
    dir_name = basedir+'match_images/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    file_name = dir_name+'match_{}_{}.jpg'.format(idx1, idx2)
    
    img_paths_1 = basedir+'undistort_images/'+"frame_{0}.jpg".format(idx1)
    img_paths_2 = basedir+'undistort_images/'+"frame_{0}.jpg".format(idx2)
    img1 = cv2.imread(img_paths_1)
    img2 = cv2.imread(img_paths_2)
    vis = np.concatenate((img1, img2), axis=1)

    for match in matches:
        pt_src = (int(keypoints_1[int(match[1])].x()), int(
            keypoints_1[int(match[1])].y()))
        pt_dst = (int(keypoints_2[int(match[3])].x()+640),
                  int(keypoints_2[int(match[3])].y()))

        cv2.circle(vis, pt_src, 3, (0, 255, 0), -1, lineType=16)
        cv2.circle(vis, pt_dst, 3, (0, 255, 0), -1, lineType=16)
        cv2.line(vis, pt_src, pt_dst, (255, 0, 255), 1)
    cv2.imwrite(file_name, vis)


def visualize_matched_features(basedir, number_images):
    """Visualize matched features.
        Input:
            basedir - string, the base directory of the all images.
            number_images - int
    """
    for idx1 in range(0, number_images-1):
        for idx2 in range(idx1+1, number_images):
            print("Generate {0} and {1} frame match feature visualization.".format(idx1, idx2))
            matches = read_matches(basedir, idx1, idx2)
            keypoints_1, _ = read_features(basedir, idx1)
            keypoints_2, _ = read_features(basedir, idx2)
            save_match_images(basedir, idx1, idx2, matches,
                              keypoints_1, keypoints_2)
