"""Parser."""


import numpy as np

from shonan_averaging.myconfig import *
from utilities.pose_estimate_generator import pose_estimate_generator_rectangle


def read_essential_data():
    """Read Essential Data from file."""
    dir_name = basedir+'matches/'
    file_name = dir_name+'essential_matrices.dat'

    with open(file_name) as file:
        lines = file.readlines()[2:]
    data = [list(map(np.float,match.split())) for match in lines]
    essential_dict = {(int(match[0]),int(match[1])):np.array(match[2:11]).reshape(3,3) for match in data}
    
    return essential_dict

def generate_g20_data_file():
    pass

def decompose_essential():
    """Decompose essential matrices into rotationa and transition."""

    # Create initial estimation.
    pose_estimates = pose_estimate_generator_rectangle(
            theta, delta_x, delta_y, delta_z, prior1_delta, prior2_delta, rows, cols, angles)
    # Get essential matrices
    essential_dict = read_essential_data()
    # Decompose essential matrices
    for match in essential_dict:


    pass


decompose_essential()
