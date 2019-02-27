"""
The file is used to generate unittest data for trajectory estimator

Steps of Generating unittest dataset for trajectory estimator
1. define coordinates, world coordinate is (x,y,z) and x axis is pointing forward, the camera coordinate is (z,x,-y)
2. design a 5 poses trajectory
3. Back Project the four image corners on to a wall, which is 10m along the x axis, to find the maximum area that points can be project to the image
4. Create 10 points for the map
5. Give each point a descriptor
6. Project all points in map to different images
7. Each feature point will need a descriptor

There should be two classes:
Map: 
    * feature_points
    * descriptors 3*1 numpy array

Trajectory:
    * each camera pose
    * superpoint descriptors
    * superpoint feature points for each pose
"""

import gtsam
from gtsam import Point2, Point3, Pose3
import numpy as np


class Features(object):
    def __init__(self, key_points, descriptors):
        self.key_points = key_points

        self.descriptors = descriptors

    def get_point(self, point_index):
        return

    def get_descriptor(self, descriptor_index):
        return

class AtriumMap(object):
    def create_atrium_map(self):
        self.atrium_points = [
            Point3(10.0, -15.0, 15.0),
            Point3(10.0, -5.0, 10.),
            Point3(10.0, 10.0, 5.0),
            Point3(10.0, 15.0, 0.0),
            Point3(10.0, 20.0, -5.0),
            Point3(10.0, -22.0, 15.0),
            Point3(10.0, -10.0, 20.0),
            Point3(10.0, -10.0, 10.0),
            Point3(10.0, 31.0, 17.0),
            Point3(10.0, 35.0, 15.0)]

        self.atrium_descriptors = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [
                                           0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [0.7, 0.7, 0.7], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9], [1.1, 1.1, 1.1]])

def create_superpoints():

    return

class Trajectory(object):

    def __init__(self):
        self.calibration = gtsam.Cal3_S2(128, 640, 480)
        self.key_point_list = []

    def create_poses(self):
        # Create the set of ground-truth poses
        poses = []
        vision_area_list = []
        vision_area = []
        for i, y in enumerate([0, 2.5, 5, 7.5, 10]):
            wRc = gtsam.Rot3(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]).T)
            poses.append(gtsam.Pose3(wRc, gtsam.Point3(0, y, 1.5)))

            point_1 = self.back_project(Point2(0, 0), self.calibration, 10, y)
            vision_area.append(point_1)
            point_2 = self.back_project(
                Point2(640, 0), self.calibration, 10, y)
            vision_area.append(point_2)
            point_3 = self.back_project(
                Point2(640, 480), self.calibration, 10, y)
            vision_area.append(point_3)
            point_4 = self.back_project(
                Point2(0, 480), self.calibration, 10, y)
            vision_area.append(point_4)

            vision_area_list.append(vision_area)
            vision_area = []
        print(vision_area_list)

        return poses

    def back_project(self, feature_point, calibration, depth, y):
        """back-project to 3D point at given depth, in camera coordinates."""
        pn = self.calibration.calibrate(feature_point)  # normalized
        return gtsam.Point3(depth, y+depth*(pn.x()), 1.5-(pn.y())*depth)


if __name__ == "__main__":
    trajectory = Trajectory()
    poses = Trajectory.create_poses()
    print(poses)
