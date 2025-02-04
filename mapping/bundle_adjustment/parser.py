"""Module for parsing with feature extraction and feature matching data file."""
# cSpell: disable=invalid-name

# from collections import defaultdict

import numpy as np

import gtsam


def parse_matches(data, skip_lines=0):
    """
    Parse the data to get the matches as a list of [frame_1, keypt_1, frame_2, keypt_2]

    Return:
        frames: The frames across which the matches exist
        matches: List of matches
    """
    data = data[skip_lines:]
    num_landmarks = list(map(int, data[0].split(' ')))
    num_matches = int(data[1])
    matches = [list(map(int, data[2+idx].split(' ')))
               for idx in range(num_matches)]

    return num_landmarks, matches


def get_matches(filename):
    """
    Get the matches from the file `filename`

    Return:
        frames: The frames across which the matches exist
        matches: List of matches
    """
    with open(filename) as file:
        lines = file.readlines()

    frames, matches = parse_matches(lines, skip_lines=4)
    return frames, matches


def load_features(filename):
    """
    Load features from feature file `filename`

    Return
        features: List of tuples with each tuple as keypoint(list) and feature descriptor(list)
    """
    with open(filename) as file:
        data = file.readlines()

    num_features, _ = list(map(int, data[0].split()))

    def extract_kp(idx):
        desc = list(map(float, data[idx].split()))
        return gtsam.Point2(desc[0], desc[1])  # pylint:disable=no-member

    def extract_desc(idx):
        desc = list(map(float, data[idx].split()))
        return desc[2:]
    skip_line = 1
    keypoints = [extract_kp(idx)
                 for idx in range(skip_line, num_features+skip_line)]
    descriptors = [extract_desc(idx) for idx in range(
        skip_line, num_features+skip_line)]

    return keypoints, descriptors


def load_features_array(filename):
    """
    Load features from feature file `filename`

    Return
        features: List of tuples with each tuple as keypoint(numpy array) and feature descriptor(numpy array)
    """
    with open(filename) as file:
        data = file.readlines()

    num_features, feature_len = list(map(int, data[0].split()))

    key_points = []
    descriptors = []
    for idx in range(1, num_features+1):
        desc = list(map(float, data[idx].split()))
        point = [desc[0], desc[1]]
        descriptor = desc[2:]

        # ensure the descriptor length is as specified
        assert len(descriptor) == feature_len
        key_points.append(point)
        descriptors.append(descriptor)
    return np.array(key_points), np.array(descriptors)


def load_features_list(filename):
    """
    Load features from feature file `filename`

    Return
        features: List of tuples with each tuple as keypoint(list) and feature descriptor(list)
    """
    with open(filename) as file:
        data = file.readlines()

    num_features, _ = list(map(int, data[0].split()))

    def extract_kp(idx):
        desc = list(map(float, data[idx].split()))
        return [desc[0], desc[1]]

    def extract_desc(idx):
        desc = list(map(float, data[idx].split()))
        return desc[2:]

    keypoints = [extract_kp(idx) for idx in range(1, num_features+1)]
    descriptors = [extract_desc(idx) for idx in range(1, num_features+1)]

    return keypoints, descriptors
