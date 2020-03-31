"""Helper functions for trajectory estimator."""
import os

import cv2
import numpy as np

# Font parameters for visualizaton.
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_CLR = (255, 255, 255)
FONT_PT = (4, 12)
FONT_SC = 0.4


def leading_zero(index):
    """Create leading zero filename"""
    index_string = str(index)
    index_len = len(index_string)
    output = ['0' for i in range(7)]
    for i in range(index_len):
        output[7-index_len+i] = index_string[i]
    return ''.join(output)


def save_feature_to_file(dir_name, features, index):
    """This is used to save the features information of each frame into a .key file,[x,y, desc(256)]"""
    nrpoints = features.get_length()
    descriptor_length = 256

    features = [np.hstack((features.keypoint(i), features.descriptor(i)))
                for i in range(nrpoints)]
    features = np.array(features)

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    np.savetxt(dir_name + leading_zero(index) +
               '.key', features)

    first_line = str(nrpoints)+' '+str(descriptor_length)+'\n'
    with open(dir_name+leading_zero(index)+'.key', 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(first_line+content)


def save_feature_image(dir_name, features, color_image, index, color=(0, 255, 0)):
    """Save image with features extracted on the image."""
    # Extra output -- Show current point detections.
    for pt in features.keypoints:
        pt = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(color_image, pt, 1, color, -1, lineType=16)
        cv2.putText(color_image, 'Raw Point Detections', FONT_PT,
                    FONT, FONT_SC, FONT_CLR, lineType=16)

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    out_file_1 = dir_name+'frame_%05d' % index+'.jpg'
    print('Writing image to %s' % out_file_1)
    cv2.imwrite(out_file_1, color_image)


def save_match_image(dir_name, observations, keypoints, color_image, index):
    """Save image with extracted features and projected features on the image."""
    # Extra output -- Show current point detections.
    for i, observation in enumerate(observations):
        pt1 = (int(round(observation[0].x())), int(round(observation[0].y())))
        cv2.circle(color_image, pt1, 5, (0, 255, 0), -1, lineType=16)
        pt2 = keypoints[i]
        pt2 = (int(round(pt2[0])), int(round(pt2[1])))
        cv2.circle(color_image, pt2, 2, (255, 0, 0), -1, lineType=16)
        cv2.line(color_image, pt1, pt2, (0, 0, 0), 1)
    cv2.putText(color_image, 'Green is extract features and Red is project features.',
                FONT_PT, FONT, FONT_SC, FONT_CLR, lineType=16)

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    out_file = dir_name+'frame_%05d' % index+'.jpg'
    print('Writing image to %s' % out_file)
    cv2.imwrite(out_file, color_image)
