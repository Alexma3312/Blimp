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
from localization.configs.myconfig_raspi_inner_degenerate import *
from mapping.bundle_adjustment.mapping_result_helper import load_poses_from_file


def run():
    """Execution."""
    directory_name = "/home/sma96/datasets/spring2020/raspi/kluas/localization/raspi_inner_degenerate/"
    poses = load_poses_from_file(directory_name+'map/poses.dat')
    # 1000
    initial_pose = [1.902099991905188414e+00, 1.656925538968248568e-01, 1.130714771245433736e+00, 3.001053052505483487e-01, 3.000391073325772706e-01, 9.054906624761668299e-01, -3.209157043353065286e-01, 9.256710093625487579e-01, -2.003653990505089855e-01, -8.983039109741298711e-01, -2.304554544738816602e-01, 3.740860422841008615e-01]
    rotation = Rot3(np.array(initial_pose[3:]).reshape(3, 3))
    initial_pose = Pose3(rotation, Point3(np.array(initial_pose[0:3])))

    l2_thresh = 1.2
    distance_thresh = [35, 35]
    trajectory_estimator = TrajectoryEstimator(
        initial_pose, directory_name, camera, l2_thresh, distance_thresh, noise_models, True, True)

    camid = 1
    skip = 1
    start_index = 0
    img_glob = "*.jpg"

    image_directory_path = directory_name+'source_images/'
    trajectory = trajectory_estimator.trajectory_generator(
        image_directory_path, camid, skip, img_glob, start_index)

    actual_poses = load_poses_from_file(directory_name+"map/poses.dat")
    # plot_trajectory(trajectory)
    # plot_trajectory_verification(
    #     trajectory_estimator.map.landmarks, actual_poses, [], 8,8,8, 0.5)

if __name__ == "__main__":
    run()
