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
from localization.configs.myconfig_raspi_inner_640x480 import *
from mapping.bundle_adjustment.mapping_result_helper import load_poses_from_file


def run():
    """Execution."""
    directory_name = "/home/sma96/datasets/spring2020/raspi/kluas/localization/raspi_inner_137_640x480_debug_rotation/"
    initial_pose = [1.656793670361359716e+00, 8.613336032728190206e-02, 1.227660966577803769e+00, 2.620000263076869840e-01, 4.419291881879974238e-01, 8.579362323869166307e-01, -3.679825873524115720e-01, 8.675750102696201393e-01, -3.345181862934931694e-01, -8.921573861265678085e-01, -2.280618209678228758e-01, 3.899269216105094027e-01]
    rotation = Rot3(np.array(initial_pose[3:]).reshape(3, 3))
    initial_pose = Pose3(rotation, Point3(np.array(initial_pose[0:3])))

    l2_thresh = 1.0
    distance_thresh = [50, 50]
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
