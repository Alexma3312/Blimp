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
from localization.configs.myconfig_raspi_137 import *
from mapping.bundle_adjustment.mapping_result_helper import load_poses_from_file


def run():
    """Execution."""
    directory_name = "/home/sma96/datasets/spring2020/raspi/kluas/localization/raspi_137/"
    poses = load_poses_from_file(directory_name+'map/poses.dat')
    # initial
    # initial_pose = poses[6]
    # 811
    # initial_pose = [-1.294592407815835733e+00, -2.772808393471121180e-01, -1.035613640087762813e-02, 9.269264269642482068e-01, 8.042938607068978764e-04, -3.752422578903907846e-01, 7.825417200031871412e-02, 9.775967139127663685e-01, 1.953994613899960597e-01, 3.669927568220409109e-01, -2.104851967677085212e-01, 9.060972896890410189e-01]

    # 509
    # initial_pose = [-1.507702963807016072e+00, -4.761252507548139890e-01, -1.994169971642082873e+00, 9.475337760652791097e-01, -9.310834764698003307e-03, 3.195200331300862007e-01, -2.918972129398649668e-02, 9.928791464233593933e-01, 1.154944187759956581e-01, -3.183201272088338474e-01, -1.187615634521963920e-01, 9.405147461152070232e-01]
    # 1895
    # initial_pose = [3.712468812530212436e-01, 8.746371347926209516e-01, 2.594915200762292429e+00, 5.191582543761233959e-01, 9.434453977343043585e-02, 8.494550104202308605e-01, -2.212810447180709461e-01, 9.748370882606341992e-01, 2.696943825783188614e-02, -8.255358297261121692e-01, -2.019696986342329470e-01, 5.269713793670129931e-01]
    # 2205
    initial_pose = [2.191608641725544704e+00, 5.674051113998604956e-01, 2.948437173559570024e+00, 1.028844154040116921e-01, -8.220067410718574774e-03, 9.946593525216054221e-01, -2.296257409560649443e-01, 9.727596626814264402e-01, 3.179084381148156863e-02, -9.678258191210414862e-01, -2.316701732024022686e-01, 9.819427015368058231e-02]
    
    rotation = Rot3(np.array(initial_pose[3:]).reshape(3, 3))
    initial_pose = Pose3(rotation, Point3(np.array(initial_pose[0:3])))

    l2_thresh = 1.5
    distance_thresh = [15, 15]
    trajectory_estimator = TrajectoryEstimator(
        initial_pose, directory_name, camera, l2_thresh, distance_thresh, noise_models, True, True)

    camid = 1
    skip = 1
    start_index = 0
    img_glob = "*.jpg"

    image_directory_path = directory_name+'source_images_debug_2205/'
    trajectory = trajectory_estimator.trajectory_generator(
        image_directory_path, camid, skip, img_glob, start_index)

    actual_poses = load_poses_from_file(directory_name+"map/poses.dat")
    # plot_trajectory(trajectory)
    # plot_trajectory_verification(
    #     trajectory_estimator.map.landmarks, actual_poses, [], 8,8,8, 0.5)

if __name__ == "__main__":
    run()
