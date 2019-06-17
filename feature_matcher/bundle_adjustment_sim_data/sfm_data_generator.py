"""
Functions to generate structure-from-motion ground truth data and sfm input data.
"""
import numpy as np
import gtsam
from gtsam import Point2, Point3, Pose3, Rot3


def create_points():
    # Create the set of ground-truth landmarks
    points = [Point3(10.0, 0.0, 0.0),
              Point3(10.0, 5.0, 0.0),
              Point3(10.0, 2.5, 2.5),
              Point3(10.0, 0.0, 5.0),
              Point3(10.0, 5.0, 5.0)]
    return points


def create_poses():
    # Create the set of ground-truth poses
    poses = []
    for y in [0, 2.5, 5]:
        # wRc is the rotation matrix that transform the World Coordinate to Camera Coordinate
        # World Coordinate (x,y,z) -> Camera Coordinate (z,x,-y)
        wRc = Rot3(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]).T)
        poses.append(Pose3(wRc, Point3(0, y, 1.5)))
    return poses


class Data(object):
    """Create input data included camera poses and landmark points for Structure from Motion. """

    def __init__(self, nrCameras, nrPoints):
        self.nrCameras = nrCameras
        self.nrPoints = nrPoints
        self.Z = [x[:]
                  for x in [[Point2()] * self.nrPoints] * self.nrCameras]
        self.J = [x[:] for x in [[0] * self.nrPoints] * self.nrCameras]

    def generate_data(self, choice):
        """All feature points are manually collected. """
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

        elif choice == 3:

            self.Z = [
                [Point2(548, 248), Point2(228, 252), Point2(184, 264), Point2(368, 440), Point2(548, 288), Point2(
                    328, 400), Point2(144, 244), Point2(216, 384), Point2(212, 324), Point2(168, 368)],
                [Point2(536, 228), Point2(216, 236), Point2(180, 244), Point2(360, 424), Point2(536, 268), Point2(
                    316, 384), Point2(136, 228), Point2(204, 368), Point2(200, 308), Point2(156, 352)],
                [Point2(492, 236), Point2(176, 244), Point2(140, 248), Point2(312, 432), Point2(492, 276), Point2(
                    268, 392), Point2(96, 236), Point2(160, 368), Point2(156, 320), Point2(112, 352)],
                [Point2(492, 228), Point2(180, 244), Point2(148, 244), Point2(316, 428), Point2(492, 272), Point2(
                    276, 388), Point2(104, 236), Point2(164, 360), Point2(164, 312), Point2(116, 348)],
                [Point2(448, 236), Point2(152, 244), Point2(124, 252), Point2(280, 432), Point2(444, 280), Point2(
                    240, 388), Point2(80, 240), Point2(132, 364), Point2(128, 312), Point2(80, 344)]
            ]

            self.J = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3,
                                                       4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

        elif choice == 4:

            self.Z = [
                [Point2(548/4, 248/4), Point2(228/4, 252/4), Point2(184/4, 264/4), Point2(368/4, 440/4), Point2(548/4, 288/4),
                 Point2(328/4, 400/4), Point2(144/4, 244/4), Point2(216/4, 384/4), Point2(212/4, 324/4), Point2(168/4, 368/4)],
                [Point2(536/4, 228/4), Point2(216/4, 236/4), Point2(180/4, 244/4), Point2(360/4, 424/4), Point2(536/4, 268/4),
                 Point2(316/4, 384/4), Point2(136/4, 228/4), Point2(204/4, 368/4), Point2(200/4, 308/4), Point2(156/4, 352/4)],
                [Point2(492/4, 236/4), Point2(176/4, 244/4), Point2(140/4, 248/4), Point2(312/4, 432/4), Point2(492/4, 276/4),
                 Point2(268/4, 392/4), Point2(96/4, 236/4), Point2(160/4, 368/4), Point2(156/4, 320/4), Point2(112/4, 352/4)],
                [Point2(492/4, 228/4), Point2(180/4, 244/4), Point2(148/4, 244/4), Point2(316/4, 428/4), Point2(492/4, 272/4),
                 Point2(276/4, 388/4), Point2(104/4, 236/4), Point2(164/4, 360/4), Point2(164/4, 312/4), Point2(116/4, 348/4)],
                [Point2(448/4, 236/4), Point2(152/4, 244/4), Point2(124/4, 252/4), Point2(280/4, 432/4), Point2(444/4, 280/4),
                 Point2(240/4, 388/4), Point2(80/4, 240/4), Point2(132/4, 364/4), Point2(128/4, 312/4), Point2(80/4, 344/4)]
            ]

            self.J = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3,
                                                       4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
