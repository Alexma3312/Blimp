"Read pose estimation result"
import numpy as np

from gtsam import Point3, Pose3, Rot3


def read_pose_estimation_result(file_path, size, sample_size):
    """
    Arguments:
        file_path: the path of the file
        start_index, end_index: start and end index
    Returns:
        poses: image poses
        invalid_indices: invalid pose indices:
    """
    # read poses 
    with open(file_path) as file:
            lines = file.readlines()
    poses = []
    for _ in range(size):
        poses.append(None)

    for idx in range(len(lines)):
        pose = list(map(float, lines[idx].split()))
        image_idx = pose[0]
        if image_idx%sample_size == 0:
            pose = pose[1:]
            rotation = Rot3(np.array(pose[3:]).reshape(3, 3))
            pose = Pose3(rotation, Point3(np.array(pose[0:3])))
            poses[int(image_idx/sample_size)] = pose
    
    bad_indices = set()
    for idx, pose in enumerate(poses):
        if pose == None:
            bad_indices.add(idx)

    return poses, bad_indices

