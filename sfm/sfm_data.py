"""
Functions to generate structure-from-motion ground truth data and iSAM input data.

"""

import numpy as np
import math
import cv2

import gtsam
from gtsam import Point2


def create_points():
    # Create the set of ground-truth landmarks
    points = [gtsam.Point3(10.0, 0.0, 0.0),
              gtsam.Point3(10.0, 5.0, 0.0),
              gtsam.Point3(10.0, 2.5, 2.5),
              gtsam.Point3(10.0, 0.0, 5.0),
              gtsam.Point3(10.0, 5.0, 5.0)]
    return points


def create_poses():
    # Create the set of ground-truth poses
    poses = []
    angle = 0

    for i, y in enumerate([-1, 0, 1]):
        theta = np.radians(-y*angle)

        # wRc is the rotation matrix that transform the World Coordinate to Camera Coordinate
        # World Coordinate (x,y,z) -> Camera Coordinate (z,x,-y)
        wRc = gtsam.Rot3(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]).T)
        # wRc = gtsam.Rot3(np.array([[0, math.cos(
        #     theta), -math.sin(theta)], [0, -math.sin(theta), -math.cos(theta)], [1, 0, 0]]).T)

        poses.append(gtsam.Pose3(wRc, gtsam.Point3(0, (y+1)*2.5, 1.5)))

    return poses


def read_images():
    Images = []
    image_1 = cv2.imread('dataset/wall_data/raw_frame_left.jpg')
    image_2 = cv2.imread('dataset/wall_data/raw_frame_middle.jpg')
    image_3 = cv2.imread('dataset/wall_data/raw_frame_right.jpg')
    Images.append(image_1)
    Images.append(image_2)
    Images.append(image_3)
    return Images


class Data(object):
    def __init__(self, nrCameras, nrPoints):
        self.nrCameras = nrCameras
        self.nrPoints = nrPoints
        self.Z = [x[:]
                  for x in [[gtsam.Point2()] * self.nrPoints] * self.nrCameras]
        self.J = [x[:] for x in [[0] * self.nrPoints] * self.nrCameras]

    def generate_data(self, choice):

        if choice == 0:
            # Generate a 5 points Map data. Points are within [160,120].
            self.Z = [[Point2(88, 63), Point2(72, 64), Point2(61, 76), Point2(82, 99), Point2(92, 98)],
                      [Point2(76, 74), Point2(60, 73), Point2(
                          46, 86), Point2(59, 110), Point2(70, 110)],
                      [Point2(86, 45), Point2(70, 42), Point2(56, 54), Point2(60, 77), Point2(70, 79)]]
            self.J = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]

        elif choice == 1:
            # Generate a 10 points Map data. Points are within [640,480].
            self.Z = [[Point2(352, 252), Point2(288, 256), Point2(244, 313), Point2(328, 396), Point2(368, 392),
                       Point2(140, 188), Point2(328, 296), Point2(456, 164), Point2(520, 224), Point2(500, 140)],
                      [Point2(304, 296), Point2(240, 292), Point2(184, 344), Point2(236, 440), Point2(280, 440),
                       Point2(108, 208), Point2(272, 340), Point2(428, 220), Point2(480, 280), Point2(464, 184)],
                      [Point2(344, 180), Point2(280, 168), Point2(224, 216), Point2(240, 308), Point2(280, 316),
                       Point2(168, 92), Point2(308, 216), Point2(484, 124), Point2(516, 200), Point2(512, 92)]]
            self.J = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3,
                                                       4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

        elif choice == 2:
            # Generate a 5 points Map data. Points are within [640,480].
            self.Z = [[Point2(352, 252), Point2(288, 256), Point2(244, 313), Point2(328, 396), Point2(368, 392)],
                      [Point2(304, 296), Point2(240, 292), Point2(
                          184, 344), Point2(236, 440), Point2(280, 440)],
                      [Point2(344, 180), Point2(280, 168), Point2(224, 216), Point2(240, 308), Point2(280, 316)]]
            self.J = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]


# Test if the functions are working
if __name__ == "__main__":
    read_images()
