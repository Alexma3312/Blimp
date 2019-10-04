"""Execution File"""
from shonan_averaging.shonan_helper import read_rotation_data,generate_rotation_edge,generate_g20_data_file
from shonan_averaging.myconfig_perpend import *

def run():
    # Create initial estimation.
    rotation_dict = read_rotation_data()
    edges = generate_rotation_edge(rotation_dict)
    generate_g20_data_file(pose_estimates, edges)

if __name__ == "__main__":
    run()