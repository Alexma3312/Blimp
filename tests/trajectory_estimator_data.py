"""
The file is used to generate unittest data for trajectory estimator unittests.

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
    * landmark points valide projected features
    * superpoint features
    * matched feature points
    * visible_map
"""

import gtsam
from gtsam import Point2, Point3, Pose3
import numpy as np

from atrium_control.mapping_and_localization_data import Map, Features


def create_atrium_map():
    """
    Create a atrium map with 10 landmark points. Create 3 dimensional descriptors for each landmark.
    """
    atrium_points = [
        # (x,y,z)
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

    atrium_descriptors = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[1,1,0,0,1,0],[1,1,1,0,1,0],[0,0,0,0,0,1],[1,1,1,1,1,0],[0,0,0,0,1,1],])

    atrium_map = Map(np.array(atrium_points), np.array(atrium_descriptors))

    return atrium_map


class Trajectory(object):
    """
    A trajectory contains all the poses of the camera.
    Each pose also contain information of:
        1. the feature points of valid projected landmark points
        2. Based on the valid feature points, we create artifact superpoint point
        3. back project visual area
        4. matched feature points
        5. visible_map
    """

    def __init__(self, fov, width, height, nr_points):
        """
        Key Arguments:
            calibration - the calibration of a camera
            poses - a list of the camera poses, each pose is a gtsam.Pose3 object
            vision_area_list - a list of vision area of each pose, back project the vertices of the image to distance 10m to get its vision area at 10m
            projected_features - a list of features, each element is a collection of features which are landmarks that projected to the current pose
            superpoint_features - 
            match_features - 
            visible_map -   
        """
        self.calibration = gtsam.Cal3_S2(fov, width, height)
        self.poses = []
        self.past_poses = []
        self.vision_area_list = []
        self.projected_features = []
        self.superpoint_features = []
        self.matched_features = []
        self.visible_map = []

    def back_project(self, feature_point, calibration, depth, y):
        """back-project to 3D point at given depth, in camera coordinates."""
        pn = self.calibration.calibrate(feature_point)  # normalized
        return gtsam.Point3(depth, y+depth*(pn.x()), 1.5-(pn.y())*depth)

    def create_past_poses(self, delta):
        # Create the set of ground-truth poses
        for i, y in enumerate([0, 2.5, 5, 7.5, 10]):
            wRc = gtsam.Rot3(np.array([[0+delta, 1+delta, 0+delta], [0+delta, 0-delta, -1+delta], [1-delta, 0+delta, 0-delta]]).T)
            self.past_poses.append(gtsam.Pose3(wRc, gtsam.Point3(0, y-delta, 1.5)))

    def create_poses(self):
        # Create the set of ground-truth poses
        for i, y in enumerate([0, 2.5, 5, 7.5, 10]):
            wRc = gtsam.Rot3(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]).T)
            self.poses.append(gtsam.Pose3(wRc, gtsam.Point3(0, y, 1.5)))

    def calculate_vision_area(self):
        for y in [0, 2.5, 5, 7.5, 10]:
            vision_area = []
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

            self.vision_area_list.append(vision_area)

    def create_project_features(self):
        # 0,1,2,3,4,7
        pose0_points = [Point2(85.8884, 29.2995), Point2(241.963, 107.337), Point2(476.074, 185.374), Point2(554.112, 263.411), Point2(632.149, 341.448), Point2(163.926, 107.337)]
        pose0_descriptors = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
        pose0_features = Features(
            np.array(pose0_points), np.array(pose0_descriptors))
        self.projected_features.append(pose0_features)

        # 0,1,2,3,4,7
        pose1_points = [Point2(46.8698, 29.2995), Point2(202.944, 107.337), Point2(437.056, 185.374), Point2(515.093, 263.411), Point2(593.13, 341.448), Point2(124.907, 107.337)]
        pose1_descriptors = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
        pose1_features = Features(
            np.array(pose1_points), np.array(pose1_descriptors))
        self.projected_features.append(pose1_features)

        # 0,1,2,3,4,7
        pose2_points = [Point2(7.85114, 29.2995), Point2(163.926, 107.337), Point2(398.037, 185.374), Point2(476.074, 263.411), Point2(554.112, 341.448), Point2(85.8884, 107.337)]
        pose2_descriptors = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
        pose2_features = Features(
            np.array(pose2_points), np.array(pose2_descriptors))
        self.projected_features.append(pose2_features)

        # 1,2,3,4,7
        pose3_points = [Point2(124.907, 107.337), Point2(359.019, 185.374), Point2(437.056, 263.411), Point2(515.093, 341.448), Point2(46.8698, 107.337)]
        pose3_descriptors = [[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
        pose3_features = Features(
            np.array(pose3_points), np.array(pose3_descriptors))
        self.projected_features.append(pose3_features)

        # 1,2,3,4,7
        pose4_points = [Point2(85.8884, 107.337), Point2(320, 185.374), Point2(398.037, 263.411), Point2(476.074, 341.448), Point2(7.85114, 107.337)]
        pose4_descriptors = [[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
        pose4_features = Features(
            np.array(pose4_points), np.array(pose4_descriptors))
        self.projected_features.append(pose4_features)

    def create_superpoint_features(self):
        """
        Superpoint feature points are all integer
        """
        # 0* 1,2,3,4
        pose0_points = [Point2(15, 29), Point2(241, 107), Point2(
            476, 185), Point2(554, 263), Point2(632, 341)]
        pose0_descriptors = [[0,1,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0]]
        pose0_features = Features(
            np.array(pose0_points), np.array(pose0_descriptors))
        self.superpoint_features.append(pose0_features)

        # 1, 2*, 3,4,7
        pose1_points = [Point2(65, 107), Point2(437, 185), Point2(
            515, 263), Point2(593, 341), Point2(124, 107)]
        pose1_descriptors = [[0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
        pose1_features = Features(
            np.array(pose1_points), np.array(pose1_descriptors))
        self.superpoint_features.append(pose1_features)

        # 0,1,3,4,7
        pose2_points = [Point2(7, 29), Point2(163, 107), Point2(
            476, 263), Point2(554, 341), Point2(85, 107)]
        pose2_descriptors = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
        pose2_features = Features(
            np.array(pose2_points), np.array(pose2_descriptors))
        self.superpoint_features.append(pose2_features)

        # 1,2,3,7
        pose3_points = [Point2(124, 107), Point2(
            359, 185), Point2(437, 263), Point2(46, 107)]
        pose3_descriptors = [[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,1]]
        pose3_features = Features(
            np.array(pose3_points), np.array(pose3_descriptors))
        self.superpoint_features.append(pose3_features)

        # 1,2,3,4
        pose4_points = [Point2(85, 107), Point2(
            320, 185), Point2(398, 263), Point2(476, 341)]
        pose4_descriptors = [[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0]]
        pose4_features = Features(
            np.array(pose4_points), np.array(pose4_descriptors))
        self.superpoint_features.append(pose4_features)

    def create_matched_features(self):
        # 1,2,3,4
        pose0_points = [Point2(241, 107), Point2(
            476, 185), Point2(554, 263), Point2(632, 341)]
        pose0_descriptors = [[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0]]
        pose0_features = Features(
            np.array(pose0_points), np.array(pose0_descriptors))
        self.matched_features.append(pose0_features)

        # 1,3,4,7
        pose1_points = [Point2(65, 107), Point2(
            515, 263), Point2(593, 341), Point2(124, 107)]
        pose1_descriptors = [[0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
        pose1_features = Features(
            np.array(pose1_points), np.array(pose1_descriptors))
        self.matched_features.append(pose1_features)

        # 0,1,3,4,7
        pose2_points = [Point2(7, 29), Point2(163, 107), Point2(
            476, 263), Point2(554, 341), Point2(85, 107)]
        pose2_descriptors = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
        pose2_features = Features(
            np.array(pose2_points), np.array(pose2_descriptors))
        self.matched_features.append(pose2_features)

        # 1,2,3,7
        pose3_points = [Point2(124, 107), Point2(
            359, 185), Point2(437, 263), Point2(46, 107)]
        pose3_descriptors = [[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,1]]
        pose3_features = Features(
            np.array(pose3_points), np.array(pose3_descriptors))
        self.matched_features.append(pose3_features)

        # 1,2,3,4
        pose4_points = [Point2(85, 107), Point2(
            320, 185), Point2(398, 263), Point2(476, 341)]
        pose4_descriptors = [[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0]]
        pose4_features = Features(
            np.array(pose4_points), np.array(pose4_descriptors))
        self.matched_features.append(pose4_features)

    def create_visible_map(self):
        # 1,2,3,4
        pose0_points = [
            # (x,y,z)
            Point3(10.0, -5.0, 10.),
            Point3(10.0, 10.0, 5.0),
            Point3(10.0, 15.0, 0.0),
            Point3(10.0, 20.0, -5.0)]

        pose0_descriptors = [[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0]]
        
        pose0_landmarks = Map(
            np.array(pose0_points), np.array(pose0_descriptors))
        self.visible_map.append(pose0_landmarks)

        # 1,3,4,7
        pose1_points = [
            # (x,y,z)
            Point3(10.0, -5.0, 10.),
            Point3(10.0, 15.0, 0.0),
            Point3(10.0, 20.0, -5.0),
            Point3(10.0, -10.0, 10.0)]

        pose1_descriptors = [[0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]

        pose1_landmarks = Map(
            np.array(pose1_points), np.array(pose1_descriptors))
        self.visible_map.append(pose1_landmarks)

        # 0,1,3,4,7
        pose2_points = [
            # (x,y,z)
            Point3(10.0, -15.0, 15.0),
            Point3(10.0, -5.0, 10.),
            Point3(10.0, 15.0, 0.0),
            Point3(10.0, 20.0, -5.0),
            Point3(10.0, -10.0, 10.0)]

        pose2_descriptors = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]

        pose2_landmarks = Map(
            np.array(pose2_points), np.array(pose2_descriptors))
        self.visible_map.append(pose2_landmarks)

        # 1,2,3,7
        pose3_points = [
            # (x,y,z)
            Point3(10.0, -5.0, 10.),
            Point3(10.0, 10.0, 5.0),
            Point3(10.0, 15.0, 0.0),
            Point3(10.0, -10.0, 10.0)]

        pose3_descriptors = [[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,1]]

        pose3_landmarks = Map(
            np.array(pose3_points), np.array(pose3_descriptors))
        self.visible_map.append(pose3_landmarks)

        # 1,2,3,4
        pose4_points = [
            Point3(10.0, -5.0, 10.),
            Point3(10.0, 10.0, 5.0),
            Point3(10.0, 15.0, 0.0),
            Point3(10.0, 20.0, -5.0)]

        pose4_descriptors = [[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0]]

        pose4_landmarks = Map(
            np.array(pose4_points), np.array(pose4_descriptors))
        self.visible_map.append(pose4_landmarks)

if __name__ == "__main__":
    Trajectory = Trajectory(128, 640, 480, 5)
    
    Trajectory.create_poses()
    # print(Trajectory.poses)

    Trajectory.calculate_vision_area()
    # print(Trajectory.vision_area_list)

    Trajectory.create_project_features()
    # for i,pose in enumerate(Trajectory.projected_features):
    #     print("pose",i,"\n",pose.key_points)
    #     print("pose",i,"\n",pose.descriptors)
    
    Trajectory.create_superpoint_features()
    # for i,pose in enumerate(Trajectory.superpoint_features):
    #     print("pose",i,"\n",pose.key_points)
    #     print("pose",i,"\n",pose.descriptors)

    Trajectory.create_matched_features()
    # for i,pose in enumerate(Trajectory.matched_features):
    #     print("pose",i,"\n",pose.key_points)
    #     print("pose",i,"\n",pose.descriptors)

    Trajectory.create_visible_map()
    # for i,pose in enumerate(Trajectory.visible_map):
    #     print("pose",i,"\n",pose.landmark_points)
    #     print("pose",i,"\n",pose.descriptors)

    atrium_map = create_atrium_map()
    # print(atrium_map.landmark_points)
    # print(atrium_map.descriptors)
