# cSpell: disable=invalid-name
"""Pose estimate generator."""
# pylint: disable=no-name-in-module
import math

import numpy as np

from gtsam import Point3, Pose3, Rot3


def pose_estimate_generator(theta, delta_x, delta_y, delta_z, prior1_delta, prior2_delta, rows, cols, angles, wRw2=np.identity(3)):
    """Generate pose estimates for mapping.
        Parameters:
            theta - int, angle of rotation
            delta_x - int, delta distance along x axis
            delta_y - int, delta distance along y axis
            delta_z - int, height
            prior1_delta - list, [x,y,z,rotation angle]
            prior2_delta - list, [x,y,z,rotation angle]
            rows - int, number of rows, along x axis
            cols - int, number of columns, along y axis
            angles - int, number of angles
            wRw2 - numpy array, rotation of the pose estimate grid
                np.array([[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0], [math.sin(theta), 0, math.cos(theta)]])
    """
    # Camera to world rotation
    wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0).matrix()
    wRc2 = Rot3(np.dot(wRw2, wRc)).matrix()

    def image_pose(i):
        y_idx = i // (cols*angles)
        x_idx = (i % (cols*angles))//angles
        angle_idx = (i % (cols*angles)) % angles
        # Basic rotation along z axis
        degree = math.radians(theta*angle_idx)
        Rz = np.array([[math.cos(degree), -math.sin(degree), 0],
                       [math.sin(degree), math.cos(degree), 0], [0, 0, 1]])
        rotation = Rot3(np.dot(Rz, wRc2))
        translation = Point3(delta_x*x_idx, delta_y*y_idx, delta_z)
        image_pose = Pose3(rotation, translation)
        return image_pose

    image_poses = [image_pose(i) for i in range(rows*cols*angles)]

    # prior1_delta = [x_delta, y_delta, z_delta, theta_delta]
    degree_1 = math.radians(prior1_delta[3])
    prior_rot_1 = Rot3(np.dot(np.array([[math.cos(degree_1), -math.sin(degree_1), 0], [
                       math.sin(degree_1), math.cos(degree_1), 0], [0, 0, 1]]), wRc))
    prior_1 = Pose3(prior_rot_1, Point3(
        prior1_delta[0], prior1_delta[1], prior1_delta[2]))
    # prior2_delta = [x_delta, y_delta, z_delta, theta_delta]
    degree_2 = math.radians(prior2_delta[3])
    prior_rot_2 = Rot3(np.dot(np.array([[math.cos(degree_2), -math.sin(degree_2), 0], [
                       math.sin(degree_2), math.cos(degree_2), 0], [0, 0, 1]]), wRc))
    prior_2 = Pose3(prior_rot_2, Point3(
        prior2_delta[0], prior2_delta[1], prior2_delta[2]))

    image_poses[:0] = [prior_2]
    image_poses[:0] = [prior_1]

    return image_poses

def pose_estimate_generator_rectangle(theta, delta_x, delta_y, delta_z, prior1_delta, prior2_delta, rows, cols, angles, wRw2=np.identity(3)):
    """Generate pose estimates for mapping.
        Parameters:
            theta - int, angle of rotation
            delta_x - int list, delta distance along x axis
            delta_y - int list, delta distance along y axis
            delta_z - int, height
            prior1_delta - list, [x,y,z,rotation angle]
            prior2_delta - list, [x,y,z,rotation angle]
            rows - int, number of rows, along x axis
            cols - int, number of columns, along y axis
            angles - int, number of angles
            wRw2 - numpy array, rotation of the pose estimate grid
                np.array([[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0], [math.sin(theta), 0, math.cos(theta)]])
    """
    # Camera to world rotation
    wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0).matrix()
    wRc2 = Rot3(np.dot(wRw2, wRc)).matrix()

    def image_pose(i):
        y_idx = i // (cols*angles)
        x_idx = (i % (cols*angles))//angles
        angle_idx = (i % (cols*angles)) % angles
        # Basic rotation along z axis
        degree = math.radians(theta*angle_idx)
        Rz = np.array([[math.cos(degree), -math.sin(degree), 0],
                       [math.sin(degree), math.cos(degree), 0], [0, 0, 1]])
        rotation = Rot3(np.dot(Rz, wRc2))
        translation = Point3(delta_x[x_idx], delta_y[y_idx], delta_z)
        image_pose = Pose3(rotation, translation)
        return image_pose

    image_poses = [image_pose(i) for i in range(rows*cols*angles)]

    # prior1_delta = [x_delta, y_delta, z_delta, theta_delta]
    degree_1 = math.radians(prior1_delta[3])
    prior_rot_1 = Rot3(np.dot(np.array([[math.cos(degree_1), -math.sin(degree_1), 0], [
                       math.sin(degree_1), math.cos(degree_1), 0], [0, 0, 1]]), wRc))
    prior_1 = Pose3(prior_rot_1, Point3(
        prior1_delta[0], prior1_delta[1], prior1_delta[2]))
    # prior2_delta = [x_delta, y_delta, z_delta, theta_delta]
    degree_2 = math.radians(prior2_delta[3])
    prior_rot_2 = Rot3(np.dot(np.array([[math.cos(degree_2), -math.sin(degree_2), 0], [
                       math.sin(degree_2), math.cos(degree_2), 0], [0, 0, 1]]), wRc))
    prior_2 = Pose3(prior_rot_2, Point3(
        prior2_delta[0], prior2_delta[1], prior2_delta[2]))

    image_poses[:0] = [prior_2]
    image_poses[:0] = [prior_1]

    return image_poses

def pose_estimate_generator_half_circle(theta, delta_x, delta_y, delta_z, prior1_delta, prior2_delta, rows, cols, angles, wRw2=np.identity(3)):
    """Generate pose estimates for mapping.
        Parameters:
            theta - int, angle of rotation
            delta_x - int list, delta distance along x axis
            delta_y - int list, delta distance along y axis
            delta_z - int, height
            prior1_delta - list, [x,y,z,rotation angle]
            prior2_delta - list, [x,y,z,rotation angle]
            rows - int, number of rows, along x axis
            cols - int, number of columns, along y axis
            angles - int, number of angles
            wRw2 - numpy array, rotation of the pose estimate grid
                np.array([[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0], [math.sin(theta), 0, math.cos(theta)]])
    """
    # Camera to world rotation
    wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0).matrix()
    wRc2 = Rot3(np.dot(wRw2, wRc)).matrix()

    def image_pose(i):
        y_idx = i // (cols*angles)
        x_idx = (i % (cols*angles))//angles
        angle_idx = (i % (cols*angles)) % angles
        # Basic rotation along z axis
        if angle_idx <3:
            degree = math.radians(theta*angle_idx)
            Rz = np.array([[math.cos(degree), -math.sin(degree), 0],
                        [math.sin(degree), math.cos(degree), 0], [0, 0, 1]])
        if angle_idx >=3:
            degree = math.radians(-theta*(5-angle_idx))
            Rz = np.array([[math.cos(degree), -math.sin(degree), 0],
                        [math.sin(degree), math.cos(degree), 0], [0, 0, 1]])
        rotation = Rot3(np.dot(Rz, wRc2))
        translation = Point3(delta_x[x_idx], delta_y[y_idx], delta_z)
        image_pose = Pose3(rotation, translation)
        return image_pose

    image_poses = [image_pose(i) for i in range(rows*cols*angles)]

    # prior1_delta = [x_delta, y_delta, z_delta, theta_delta]
    degree_1 = math.radians(prior1_delta[3])
    prior_rot_1 = Rot3(np.dot(np.array([[math.cos(degree_1), -math.sin(degree_1), 0], [
                       math.sin(degree_1), math.cos(degree_1), 0], [0, 0, 1]]), wRc))
    prior_1 = Pose3(prior_rot_1, Point3(
        prior1_delta[0], prior1_delta[1], prior1_delta[2]))
    # prior2_delta = [x_delta, y_delta, z_delta, theta_delta]
    degree_2 = math.radians(prior2_delta[3])
    prior_rot_2 = Rot3(np.dot(np.array([[math.cos(degree_2), -math.sin(degree_2), 0], [
                       math.sin(degree_2), math.cos(degree_2), 0], [0, 0, 1]]), wRc))
    prior_2 = Pose3(prior_rot_2, Point3(
        prior2_delta[0], prior2_delta[1], prior2_delta[2]))

    image_poses[:0] = [prior_2]
    image_poses[:0] = [prior_1]

    return image_poses


def pose_estimate_generator_quad(theta, delta_x, delta_y, delta_z, prior1_delta, prior2_delta, rows, cols, angles, wRw2=np.identity(3)):
    """Generate pose estimates for mapping.
        Parameters:
            theta - int, angle of rotation
            delta_x - int list, delta distance along x axis
            delta_y - int list, delta distance along y axis
            delta_z - int, height
            prior1_delta - list, [x,y,z,rotation angle]
            prior2_delta - list, [x,y,z,rotation angle]
            rows - int, number of rows, along x axis
            cols - int, number of columns, along y axis
            angles - int, number of angles
            wRw2 - numpy array, rotation of the pose estimate grid
                np.array([[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0], [math.sin(theta), 0, math.cos(theta)]])
    """
    # Camera to world rotation
    wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0).matrix()
    wRc2 = Rot3(np.dot(wRw2, wRc)).matrix()

    def image_pose(i):
        y_idx = i // (cols*angles)
        x_idx = (i % (cols*angles))//angles
        angle_idx = (i % (cols*angles)) % angles
        # Basic rotation along z axis
        if angle_idx == 1:
            degree = math.radians(theta*angle_idx)
            Rz = np.array([[math.cos(degree), -math.sin(degree), 0],
                        [math.sin(degree), math.cos(degree), 0], [0, 0, 1]])
        if angle_idx == 2:
            degree = math.radians(-theta*angle_idx)
            Rz = np.array([[math.cos(degree), -math.sin(degree), 0],
                        [math.sin(degree), math.cos(degree), 0], [0, 0, 1]])
        rotation = Rot3(np.dot(Rz, wRc2))
        translation = Point3(delta_x[x_idx], delta_y[y_idx], delta_z)
        image_pose = Pose3(rotation, translation)
        return image_pose

    image_poses = [image_pose(i) for i in range(rows*cols*angles)]

    # prior1_delta = [x_delta, y_delta, z_delta, theta_delta]
    degree_1 = math.radians(prior1_delta[3])
    prior_rot_1 = Rot3(np.dot(np.array([[math.cos(degree_1), -math.sin(degree_1), 0], [
                       math.sin(degree_1), math.cos(degree_1), 0], [0, 0, 1]]), wRc))
    prior_1 = Pose3(prior_rot_1, Point3(
        prior1_delta[0], prior1_delta[1], prior1_delta[2]))
    # prior2_delta = [x_delta, y_delta, z_delta, theta_delta]
    degree_2 = math.radians(prior2_delta[3])
    prior_rot_2 = Rot3(np.dot(np.array([[math.cos(degree_2), -math.sin(degree_2), 0], [
                       math.sin(degree_2), math.cos(degree_2), 0], [0, 0, 1]]), wRc))
    prior_2 = Pose3(prior_rot_2, Point3(
        prior2_delta[0], prior2_delta[1], prior2_delta[2]))

    image_poses[:0] = [prior_2]
    image_poses[:0] = [prior_1]

    return image_poses