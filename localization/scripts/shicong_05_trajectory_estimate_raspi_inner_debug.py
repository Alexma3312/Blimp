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
from localization.configs.myconfig_raspi_inner_debug import *
from mapping.bundle_adjustment.mapping_result_helper import load_poses_from_file


def run():
    """Execution."""
    directory_name = "/home/sma96/datasets/spring2020/raspi/kluas/localization/raspi_inner_debug/"
    poses = load_poses_from_file(directory_name+'map/poses.dat')
    # initial
    # initial_pose = poses[5]
    # 999
    initial_pose = [1.766335876621073053e+00, 1.289008295837021367e-01, 1.211558428509138530e+00, 4.195097923857595834e-01, 2.670539063999178842e-01, 8.675792443166172596e-01, -2.274890890566308443e-01, 9.561762107651000653e-01, -1.843251700855094710e-01, -8.787833910882816291e-01, -1.200385981399861857e-01, 4.618771335581988713e-01]

    rotation = Rot3(np.array(initial_pose[3:]).reshape(3, 3))
    initial_pose = Pose3(rotation, Point3(np.array(initial_pose[0:3])))

    l2_thresh = 0.7
    distance_thresh = [60, 60]
    trajectory_estimator = TrajectoryEstimator(
        initial_pose, directory_name, camera, l2_thresh, distance_thresh, noise_models, True, True)

    camid = 1
    skip = 1
    start_index = 0
    img_glob = "*.jpg"

    image_directory_path = directory_name+'debug_descriptor_matching/'
    trajectory = trajectory_estimator.trajectory_generator(
        image_directory_path, camid, skip, img_glob, start_index)

    actual_poses = load_poses_from_file(directory_name+"map/poses.dat")
    # plot_trajectory(trajectory)
    # plot_trajectory_verification(
    #     trajectory_estimator.map.landmarks, actual_poses, [], 8,8,8, 0.5)

if __name__ == "__main__":
    run()
