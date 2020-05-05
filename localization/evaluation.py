"""Evaluation"""
import numpy as np

from gtsam import Point3, Pose3, Rot3
from mapping.bundle_adjustment.mapping_result_helper import \
    load_poses_from_file
from utilities.plotting import plot_trajectory
from tabulate import tabulate

# evaluation_directory = '/home/sma96/datasets/spring2020/raspi/kluas/localization/results/96_10fps_result'
evaluation_directory = '/home/sma96/datasets/spring2020/raspi/kluas/localization/raspi_inner/debug'
pose_file = evaluation_directory+'/poses.dat'
table_file = evaluation_directory+'/table.dat'

# read poses 
with open(pose_file) as file:
        lines = file.readlines()
poses = []
count = 0
for idx in range(len(lines)):
    pose = list(map(float, lines[idx].split()))
    pose = pose[1:]
    rotation = Rot3(np.array(pose[3:]).reshape(3, 3))
    pose = Pose3(rotation, Point3(np.array(pose[0:3])))
    if count%1 == 0:
        poses.append(pose)
    count+=1
    
# Plot poses
plot_trajectory(poses, hold=True)

# read table
with open(table_file) as file:
        lines = file.readlines()
table = []
number_poses = len(lines)
undistort_time = []
superpoint_time = []
superpoint_number = []
project_time = []
project_number = []
associate_time = [] 
match_number = []
pose_error = []
pose_estimation_time = []
total_time = []
for idx in range(number_poses):
    data = list(map(float, lines[idx].split()))
    pose_id = data[0]
    undistort_time.append(data[1])
    superpoint_time.append(data[2])
    superpoint_number.append(data[3])
    project_time.append(data[4])
    project_number.append(data[5])
    associate_time.append(data[6])
    match_number.append(data[7])
    pose_error.append(data[8])
    pose_estimation_time.append(data[9])
    total_time.append(data[10])

# generate table
average_undistort_time = sum(undistort_time)/number_poses
average_superpoint_time = sum(superpoint_time)/number_poses
average_project_time = sum(project_time)/number_poses
average_associate_time = sum(associate_time)/number_poses
average_pose_estimation_time = sum(pose_estimation_time)/number_poses
average_total_time = sum(total_time)/number_poses
max_total_time = max(total_time)
min_total_time = min(total_time)

average_superpoint_number = sum(superpoint_number)/number_poses
max_superpoint_number = max(superpoint_number)
min_superpoint_number = min(superpoint_number)

average_project_number = sum(project_number)/number_poses
max_project_number = max(project_number)
min_project_number = min(project_number)

average_match_number = sum(match_number)/number_poses
max_match_number = max(match_number)
min_match_number = min(match_number)

average_pose_error = sum(pose_error)/number_poses
max_pose_error = max(pose_error)
min_pose_error = min(pose_error)

table.append(['average_undistort_time',average_undistort_time])
table.append(['average_superpoint_time',average_superpoint_time])
table.append(['average_project_time',average_project_time])
table.append(['average_associate_time',average_associate_time])
table.append(['average_pose_estimation_time',average_pose_estimation_time])
table.append(['average_total_time',average_total_time])
table.append(['max_total_time',max_total_time])
table.append(['min_total_time',min_total_time])
table.append(['average_superpoint_number',average_superpoint_number])
table.append(['max_superpoint_number',max_superpoint_number])
table.append(['min_superpoint_number',min_superpoint_number])
table.append(['average_project_number',average_project_number])
table.append(['max_project_number',max_project_number])
table.append(['min_project_number',min_project_number])
table.append(['average_match_number',average_match_number])
table.append(['max_match_number',max_match_number])
table.append(['min_match_number',min_match_number])
table.append(['average_pose_error',average_pose_error])
table.append(['max_pose_error',max_pose_error])
table.append(['min_pose_error',min_pose_error])

# sort by error and print as github
# table.sort(key=lambda row: row[-1])
print(tabulate(table, tablefmt='github'))

# sort by n and print as latex
# latex.sort(key=lambda row: row[1])
# print(tabulate(latex, tablefmt='latex', headers=[
#       'dataset', 'n', 'm', "#angle>=25", "avg error"]))



# Compare Rotation and Translation
