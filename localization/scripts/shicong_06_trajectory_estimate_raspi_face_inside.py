"""
Localization
"""
import numpy as np

# cSpell: disable
# pylint: disable=no-name-in-module,wrong-import-order, no-member,ungrouped-imports, invalid-name
from gtsam import Cal3_S2, Point3, Pose3, Rot3
from localization.trajectory_estimator import TrajectoryEstimator
from mapping.bundle_adjustment.mapping_result_helper import \
    load_poses_from_file
from utilities.plotting import plot_trajectory_verification, plot_with_result, plot_trajectory
from localization.configs.myconfig_raspi_face_inside import *
from mapping.bundle_adjustment.mapping_result_helper import load_poses_from_file


def run():
    """Execution."""
    directory_name = "/home/sma96/datasets/spring2020/raspi/kluas/localization/raspi_face_inside/"
    poses = load_poses_from_file(directory_name+'map/poses.dat')
    # initial
    initial_pose = poses[0]

    rotation = Rot3(np.array(initial_pose[3:]).reshape(3, 3))
    initial_pose = Pose3(rotation, Point3(np.array(initial_pose[0:3])))

    l2_thresh = 1.0
    distance_thresh = [30, 30]
    motion_model = "constant"
    # dift_threshold = [1000,360]
    # motion_model = "static"
    dift_threshold = [0.1,5]
    trajectory_estimator = TrajectoryEstimator(
        initial_pose, directory_name, camera, l2_thresh, distance_thresh, noise_models, True, True,motion_model=motion_model,dift_threshold=dift_threshold)

    camid = 1
    skip = 1
    start_index = 0
    img_glob = "*.jpg"

    image_directory_path = directory_name+'source_images_inner_10fps/'
    trajectory = trajectory_estimator.trajectory_generator(
        image_directory_path, camid, skip, img_glob, start_index)

    actual_poses = load_poses_from_file(directory_name+"map/poses.dat")
    # plot_trajectory(trajectory)
    # plot_trajectory_verification(
    #     trajectory_estimator.map.landmarks, actual_poses, [], 8,8,8, 0.5)

if __name__ == "__main__":
    run()
