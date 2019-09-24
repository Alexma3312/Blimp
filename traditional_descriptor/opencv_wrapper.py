"""OpenCV Wrapper for Feature Matcher."""
# For more information: https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html

from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
parser = argparse.ArgumentParser(
    description='Code for Feature Matching with FLANN tutorial.')
parser.add_argument(
    '--input1', help='Path to input image 1.', default='box.png')
parser.add_argument('--input2', help='Path to input image 2.',
                    default='box_in_scene.png')
args = parser.parse_args()
# img1 = cv.imread(cv.samples.findFile(args.input1), cv.IMREAD_GRAYSCALE)
# img2 = cv.imread(cv.samples.findFile(args.input2), cv.IMREAD_GRAYSCALE)
img1 = cv.imread('traditional_descriptor/box.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('traditional_descriptor/box_in_scene.png',
                 cv.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)
# -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
minHessian = 400
detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
# -- Step 2: Matching descriptor vectors with a FLANN based matcher
# Since SURF is a floating-point descriptor NORM_L2 is used
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
# -- Filter matches using the Lowe's ratio test
ratio_thresh = 0.7
good_matches = []
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

# matches_number = len(good_matches)
# keypoints_1 = np.array([point.pt for point in keypoints1])
# keypoints_2 = np.array([point.pt for point in keypoints2])
# matches = np.array([[match.queryIdx, match.trainIdx] for match in good_matches])
# # E, mask = cv.findEssentialMat(keypoints1, keypoints2, [np.array([[1, 0, 200], [0, 1, 100], [0, 0, 1]]), [cv.RANSAC, [0.999, [ 1, [ good_matches]]]]])
# fundamental_mat, mask = cv.findFundamentalMat(
#             keypoints_1, keypoints_2, cv.FM_RANSAC, 1, 0.99, matches)

# -- Draw matches
img_matches=np.empty(
    (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype = np.uint8)
cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches,
               img_matches, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# -- Show detected matches
cv.imshow('Good Matches', img_matches)
cv.waitKey()
