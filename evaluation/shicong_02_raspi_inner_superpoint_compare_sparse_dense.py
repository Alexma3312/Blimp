
import matplotlib.pyplot as plt

from evaluation.utils import get_distance_angle_error, ate

poses_dat_file =  "/home/sma96/Blimps/evaluation/inner_superpoint_sparse_dense/sparse_result/poses.dat"
ground_truth_image_text_file =  "/home/sma96/Blimps/evaluation/inner_superpoint_sparse_dense/ground_truth/images.txt"
colmap_image_text_file =  "/home/sma96/Blimps/evaluation/inner_superpoint_sparse_dense/sparse_map/images.txt"
number_poses = 343
sample_rate = 5
map_number_poses = 137
distances_sparse, angles_sparse = get_distance_angle_error(poses_dat_file, ground_truth_image_text_file, colmap_image_text_file, number_poses, sample_rate, map_number_poses)
distances_sparse =[distance*2.8 for distance in distances_sparse]
ate_rot_sparse = ate(angles_sparse)
ate_pos_sparse = ate(distances_sparse)
print(ate_rot_sparse,ate_pos_sparse)

poses_dat_file = "/home/sma96/Blimps/evaluation/inner_superpoint_sparse_dense/dense_result/poses.dat"
ground_truth_image_text_file =  "/home/sma96/Blimps/evaluation/inner_superpoint_sparse_dense/ground_truth/images.txt"
colmap_image_text_file =  "/home/sma96/Blimps/evaluation/inner_superpoint_sparse_dense/dense_map/images.txt"
number_poses = 343
sample_rate = 5
map_number_poses = 137
distances_dense, angles_dense = get_distance_angle_error(poses_dat_file, ground_truth_image_text_file, colmap_image_text_file, number_poses, sample_rate, map_number_poses)
distances_dense =[distance*2.8 for distance in distances_dense]
ate_rot_dense = ate(angles_dense)
ate_pos_dense = ate(distances_dense)
print(ate_rot_dense,ate_pos_dense)


frames = [i for i in range(1,344)]
fig, axs = plt.subplots(1, 2)
axs[0].plot(frames, distances_sparse, label='Superpoint Sparse')
axs[0].plot(frames, distances_dense, label='Superpoint Dense')
axs[0].legend()
axs[0].set(xlabel='Frame', ylabel='Position error (m)')
axs[1].plot(frames, angles_sparse, label='Superpoint Sparse')
axs[1].plot(frames, angles_dense, label='Superpoint Dense')
axs[1].legend()
axs[1].set(xlabel='Frame', ylabel='Orientation error ($^\circ$)')
plt.show()
# Compute translation error and rotation error 
pass
# Plot two trajectory, black is colmap result and green is estimation results that are close to colmap and red is estimation results that are far to colmap
