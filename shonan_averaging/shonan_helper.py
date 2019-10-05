"""Parser."""
# cSpell: disable
# pylint: disable=no-name-in-module
# (no-name-in-module,wrong-import-order, no-member,ungrouped-imports, invalid-name)

import numpy as np

from gtsam import Rot3


def read_rotation_data(basedir):
    """Read Relative Rotation Data from file."""
    dir_name = basedir+'matches/'
    file_name = dir_name+'rotation.dat'

    with open(file_name) as file:
        lines = file.readlines()[2:]
    data = [list(map(np.float, match.split())) for match in lines]
    rotation_dict = {(int(match[0]), int(match[1])): np.array(
        match[2:11]).reshape(3, 3) for match in data}

    return rotation_dict


def generate_rotation_edge(rotation_dict, pose_estimates):
    """Generate Rotation Edge data."""
    # Create an edge dictiionary
    edges = {}
    # Decompose essential matrices
    for value in rotation_dict.keys():
        # Get image idices
        idx1 = value[0]
        idx2 = value[1]
        # Get essential matrix
        Rij = rotation_dict.get(value)
        quat = Rot3(Rij).quaternion()
        # Switch from w,x,y,z to x,y,z,w
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        edge = np.append(pose_estimates[idx2].translation().vector(
        )-pose_estimates[idx1].translation().vector(), quat)
        edge = np.append(edge, np.array(
            [100, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 100, 0, 0, 0, 100, 0, 0, 100, 0, 100]))
        edges[(idx1, idx2)] = edge
    return edges


def read_essential_data(basedir):
    """Read Essential Data from file."""
    dir_name = basedir+'matches/'
    file_name = dir_name+'essential_matrices.dat'

    with open(file_name) as file:
        lines = file.readlines()[2:]
    data = [list(map(np.float, match.split())) for match in lines]
    essential_dict = {(int(match[0]), int(match[1])): np.array(
        match[2:11]).reshape(3, 3) for match in data}

    return essential_dict


def generate_g20_data_file(basedir, pose_estimates, factors):
    """Generate g2o data file with pose estimations and factors."""
    dir_name = basedir+'matches/'
    file_name = dir_name+'shicong.g2o'
    f = open(file_name, 'w')
    for i, pose in enumerate(pose_estimates):
        f.write("VERTEX_SE3:QUAT {}".format(i))
        translation = pose.translation().vector()
        rotation = pose.rotation().quaternion()
        # Switch from w,x,y,z to x,y,z,w
        rotation = np.array(
            [rotation[1], rotation[2], rotation[3], rotation[0]])
        for t in translation:
            f.write(" {}".format(t))
        for q in rotation:
            f.write(" {}".format(q))
        f.write("\n")
    for factor in factors.keys():
        f.write("EDGE_SE3:QUAT {} {}".format(factor[0], factor[1]))
        for x in factors.get(factor):
            f.write(" {}".format(x))
        f.write("\n")
    f.close


def read_shonan_result(basedir, file_name):
    """Read shonan result from file."""
    file_path = basedir+file_name
    with open(file_path) as file:
        lines = file.readlines()[1:]
    data = [list(map(np.float, line.split())) for line in lines]
    shonan_result = []
    for SO3 in data:
        idx = SO3[0]
        SO3 = SO3[1:]
        SO3 = Rot3(np.array(SO3).reshape(3, 3))
        shonan_result.append(SO3)

    return shonan_result
