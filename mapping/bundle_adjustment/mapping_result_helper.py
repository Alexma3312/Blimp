# cSpell: disable=invalid-name
"""Save and load mapping result."""
import os

import numpy as np


def save_map_to_file(map_result, base_directory):
    """Save map data to file.
    Parameter:
        map_result - a NX4 list, [[x,y,z,descriptor]]
        bases_directory - the directory where mapping execute.
    """
    dir_name = base_directory+'result/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    file_name = dir_name+'map.dat'
    np.savetxt(file_name, map_result)

    with open(file_name, "r+") as out_file:
        content = out_file.read()
        out_file.seek(0, 0)
        out_file.write('/* Format:\n')
        out_file.write('num_of_landmarks\n')
        out_file.write('x y z descriptor */\n')
        out_file.write(str(len(map_result))+"\n")
        out_file.write(content)
    out_file.close()


def parse_map(data, skip_lines=0):
    """Parse the map data into landmarks and descriptors."""
    data = data[skip_lines:]
    num_landmarks = int(data[0])
    landmark_points = []
    descriptors = []
    for idx in range(1, num_landmarks+1):
        landmark = list(map(float, data[idx].split()))
        landmark_points.append(landmark[:3])
        descriptors.append(landmark[3:])
    return landmark_points, descriptors


def load_map_from_file(filename):
    """Load map data from file."""
    with open(filename) as file:
        lines = file.readlines()

    landmark_points, descriptors = parse_map(lines, skip_lines=3)
    return landmark_points, descriptors


def save_poses_to_file(camera_poses, base_directory):
    """Save pose data to file."""
    # camera_poses = np.array(camera_poses)
    # print(camera_poses[0])
    dir_name = base_directory+'result/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    file_name = dir_name+'poses.dat'
    np.savetxt(file_name, camera_poses)
    with open(file_name, "r+") as out_file:
        content = out_file.read()
        out_file.seek(0, 0)
        out_file.write('/* Format:\n')
        out_file.write('num_of_landmarks\n')
        out_file.write('x y z rotation */\n')
        out_file.write(str(len(camera_poses))+"\n")
        out_file.write(content)
    out_file.close()


def parse_poses(data, skip_lines=3):
    """Parse the poses data from file."""

    data = data[skip_lines:]
    num_poses = int(data[0])
    poses = []
    for idx in range(1, num_poses+1):
        pose = list(map(float, data[idx].split()))
        poses.append(pose)
    return poses


def load_poses_from_file(filename):
    """Load poses from file."""
    with open(filename) as file:
        lines = file.readlines()

    poses = parse_poses(lines, skip_lines=3)
    return poses


# def get_average_descriptor(self,desc):
#     desc_average_list = []
#     desc_normalize_list = []

#     for i in range(self.nrPoints):
#         desc_sum = np.zeros((1,256))
#         for j in range(self.nrCameras):
#             desc_sum += desc[i][j]
#         desc_average = desc_sum/self.nrCameras
#         desc_average_list.append(desc_average)
#         desc_normalize = desc_average/np.linalg.norm(desc_average)
#         desc_normalize_list.append(desc_normalize)

#     for i,d in enumerate(desc_normalize_list):
#         for j in range(self.nrCameras):
#             dmat = np.dot(d,desc[1][j].T)
#             dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))

#     return desc_normalize_list

# def get_median_descriptor(self,desc):
#     desc_list = []
#     theta_list = []
#     for i in range(self.nrPoints):
#         desc_1 = desc[i][0]
#         for j in range(self.nrCameras):
#             desc_2 = desc[i][0]
#             cos_theta = np.dot(desc_1,desc_2)
#             theta = math.acos(cos_theta)
#             theta_list.append(theta)
#         if len(theta_list)
