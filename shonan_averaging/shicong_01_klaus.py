"""Execution File"""
from shonan_averaging.shonan_helper import  decompose_essential
from utilities.pose_estimate_generator import pose_estimate_generator_rectangle
from shonan_averaging.myconfig import *

def run():
    # Create initial estimation.
    pose_estimates = pose_estimate_generator_rectangle(
        theta, delta_x, delta_y, delta_z, prior1_delta, prior2_delta, rows, cols, angles)
    decompose_essential(10, pose_estimates)

if __name__ == "__main__":
    run()