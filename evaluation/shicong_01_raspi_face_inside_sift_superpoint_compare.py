from evaluation.read_colmap_pose import read_colmap_pose_result
from evaluation.read_pose_estimation_result import read_pose_estimation_result
from evaluation.sim3 import Similarity3
from evaluation.utils import difference, ate, get_distance_angle_error
import matplotlib.pyplot as plt


poses_dat_file = "/home/sma96/Blimps/evaluation/face_inside_sift_superpoint_compare/sift_result/poses.dat"
ground_truth_image_text_file = "/home/sma96/Blimps/evaluation/face_inside_sift_superpoint_compare/ground_truth/images.txt"
colmap_image_text_file = "/home/sma96/Blimps/evaluation/face_inside_sift_superpoint_compare/sift_map/images.txt"
number_poses = 387
sample_rate = 5
map_number_poses = 137
distances_sift, angles_sift = get_distance_angle_error(
    poses_dat_file, ground_truth_image_text_file, colmap_image_text_file, number_poses, sample_rate, map_number_poses)
distances_sift =[distance*2.8 for distance in distances_sift]
ate_rot_sift = ate(angles_sift)
ate_pos_sift = ate(distances_sift)
print(ate_rot_sift,ate_pos_sift)


poses_dat_file = "/home/sma96/Blimps/evaluation/face_inside_sift_superpoint_compare/superpoint_result/poses.dat"
ground_truth_image_text_file = "/home/sma96/Blimps/evaluation/face_inside_sift_superpoint_compare/ground_truth/images.txt"
colmap_image_text_file = "/home/sma96/Blimps/evaluation/face_inside_sift_superpoint_compare/superpoint_map/images.txt"
number_poses = 387
sample_rate = 5
map_number_poses = 137
distances_superpoint, angles_superpoint = get_distance_angle_error(
    poses_dat_file, ground_truth_image_text_file, colmap_image_text_file, number_poses, sample_rate, map_number_poses)
distances_superpoint =[distance*2.8 for distance in distances_superpoint]
ate_rot_superpoint = ate(angles_superpoint)
ate_pos_superpoint = ate(distances_superpoint)
print(ate_rot_superpoint,ate_pos_superpoint)



frames = [i for i in range(1, 388)]
fig, axs = plt.subplots(1, 2)
axs[0].plot(frames, distances_sift, label='Root-Sift')
axs[0].plot(frames, distances_superpoint, label='Superpoint')
axs[0].legend()
axs[0].set(xlabel='Frame', ylabel='Position error (m)')
axs[1].plot(frames, angles_sift, label='Root-Sift')
axs[1].plot(frames, angles_superpoint, label='Superpoint')
axs[1].legend()
axs[1].set(xlabel='Frame', ylabel='Orientation error ($^\circ$)')
# txt = "Superpoint: $\;ATE_{rot} = {%f}(^\circ)$" % ate_rot_superpoint+"$\;ATE_{pos} = {%f}(m)$" % ate_pos_superpoint + \
#     '\n'+"Sift: $\;ATE_{rot} = {%f}(^\circ)$" % ate_rot_sift + \
#     "$\;ATE_{pos} = {%f}(m)$" % ate_pos_sift
# fig.text(.5, .05, txt, ha='center')
plt.show()
# Compute translation error and rotation error
pass
# Plot two trajectory, black is colmap result and green is estimation results that are close to colmap and red is estimation results that are far to colmap
