"Read colmap pose"
import numpy as np
from scipy.spatial.transform import Rotation as R

from colmap.read_model import read_images_text
from gtsam import Point3, Pose3, Rot3


def read_colmap_pose_result(file_path, size, start_index, end_index):
    """
    Arguments:
        file_path: the path of the file
        start_index, end_index: start and end index
    Returns:
        poses: image poses
        invalid_indices: invalid pose indices:
    """
    images = read_images_text(file_path)
    images = sorted(images.values())
    # print(images[1])
    poses = []
    for _ in range(size):
        poses.append(None)

    for image in images:
        if image[0] > size:
            break
        pose_quat = np.array(
            [image[1][1], image[1][2], image[1][3], image[1][0]])
        # pose_R is R^t
        pose_rot = R.from_quat(pose_quat).as_matrix()
        pose_t = image[2]
        pose = Pose3(Rot3(pose_rot), Point3(pose_t)).inverse()
        # poses.append(pose)
        poses[image[0]-1] = pose
    ground_truth_poses = poses[start_index:end_index+1]
    bad_indices = set()
    for i,pose in enumerate(ground_truth_poses):
        if pose == None:
            bad_indices.add(i)
    return ground_truth_poses, bad_indices
