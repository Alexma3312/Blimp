"""Execution File"""
from shonan_averaging.shonan_helper import  decompose_essential
from utilities.pose_estimate_generator import pose_estimate_generator_rectangle
from shonan_averaging.myconfig_perpend import *

def run():
    # Create initial estimation.
    decompose_essential(10, pose_estimates)

if __name__ == "__main__":
    run()