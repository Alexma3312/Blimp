"""Parser."""
import os

import numpy as np

from shonan_averaging.myconfig import (basedir, calibration_matrix,
                                       number_images)


def read_essential_data():
    """Read Data from file
    """
    dir_name = basedir+'matches/'
    file_name = dir_name+'essential_matrices.dat'

    with open(file_name) as file:
        lines = file.readlines()[2:]
    data = [list(map(np.float,match.split())) for match in lines]
    essential_dict = {(int(match[0]),int(match[1])):np.array(match[2:11]).reshape(3,3) for match in data}
    
    return essential_dict


def decompose_essential():
    read_essential_data()
    pass


read_essential_data()
