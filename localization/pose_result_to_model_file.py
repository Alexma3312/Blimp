"""Evaluation"""
import os

import numpy as np

from gtsam import Point3, Pose3, Rot3
from mapping.bundle_adjustment.mapping_result_helper import \
    load_poses_from_file
from utilities.plotting import plot_trajectory


def write_images_text(poses, file_name):
    """Write poses into images.text"""

    f = open(file_name, "a")
    for idx, pose in enumerate(poses):
        image_idx = idx + 1
        qvec = pose.rotation().quaternion()
        tvec = pose.translation().vector()
        f.write("{} {} {} {} {} {} {} {} {}".format(image_idx, float(qvec[0]), float(qvec[1]), qvec[2], qvec[3], tvec[0],tvec[1],tvec[2], 1))
        f.write("\n")
    f.write("\n")
    f.close()

def poses_to_images_text(directory):
    """Read the colmap camera result and return a list of gtsam.Pose3
    Input:
        camera_file
    Output:
        poses: a list of gtsam.Pose3
    """
    # read poses 
    pose_file = directory+'/poses.dat'
    with open(pose_file) as file:
            lines = file.readlines()
    poses = []
    count = 0
    for idx in range(len(lines)):
        pose = list(map(float, lines[idx].split()))
        pose = pose[1:]
        rotation = Rot3(np.array(pose[3:]).reshape(3, 3))
        pose = Pose3(rotation, Point3(np.array(pose[0:3]))).inverse()
        if count%1 == 0:
            poses.append(pose)
        count+=1
        
    # Plot poses
    # plot_trajectory(poses, hold=True)

    model_file = directory+'/images.txt'
    write_images_text(poses, model_file)


# evaluation_directory = '/home/sma96/datasets/spring2020/raspi/kluas/localization/results/96_10fps_result'
evaluation_directory = '/home/sma96/datasets/spring2020/raspi/kluas/localization/raspi_inner_137_640x480/debug'
poses_to_images_text(evaluation_directory)