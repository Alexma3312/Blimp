
import matplotlib.pyplot as plt

from evaluation.utils import get_distance_angle_error, ate

ground_truth_image_text_file =  "/home/sma96/Blimps/evaluation/inner_superpoint_motion_model/ground_truth/images.txt"
colmap_image_text_file =  "/home/sma96/Blimps/evaluation/inner_superpoint_motion_model/map/images.txt"
number_poses = 115
sample_rate = 15
map_number_poses = 136

############ static
poses_dat_file =  "/home/sma96/Blimps/evaluation/inner_superpoint_motion_model/static_result/poses.dat"
distances_static, angles_static = get_distance_angle_error(poses_dat_file, ground_truth_image_text_file, colmap_image_text_file, number_poses, sample_rate, map_number_poses)
distance_static =[distance*2.47 for distance in distances_static]
ate_rot_static = ate(angles_static)
ate_pos_static = ate(distances_static)
print(ate_rot_static,ate_pos_static)

############ constant
poses_dat_file =  "/home/sma96/Blimps/evaluation/inner_superpoint_motion_model/constant_reuslt/poses.dat"
distances_constant, angles_constant = get_distance_angle_error(poses_dat_file, ground_truth_image_text_file, colmap_image_text_file, number_poses, sample_rate, map_number_poses)
distances_constant =[distance*2.47 for distance in distances_constant]
ate_rot_constant = ate(angles_constant)
ate_pos_constant = ate(distances_constant)
print(ate_rot_constant,ate_pos_constant)


############ static threshold
poses_dat_file =  "/home/sma96/Blimps/evaluation/inner_superpoint_motion_model/static_threshold_result/poses.dat"
distances_static_threshold, angles_static_threshold = get_distance_angle_error(poses_dat_file, ground_truth_image_text_file, colmap_image_text_file, number_poses, sample_rate, map_number_poses)
distances_static_threshold =[distance*2.47 for distance in distances_static_threshold]
ate_rot_static_threshold = ate(angles_static_threshold)
ate_pos_static_threshold = ate(distances_static_threshold)
print(ate_rot_static_threshold,ate_pos_static_threshold)


############ constant threshold
poses_dat_file =  "/home/sma96/Blimps/evaluation/inner_superpoint_motion_model/constant_threshold_result/poses.dat"
distances_constant_threshold, angles_constant_threshold = get_distance_angle_error(poses_dat_file, ground_truth_image_text_file, colmap_image_text_file, number_poses, sample_rate, map_number_poses)
distances_constant_threshold =[distance*2.47 for distance in distances_constant_threshold]
ate_rot_constant_threshold = ate(angles_constant_threshold)
ate_pos_constant_threshold = ate(distances_constant_threshold)
print(ate_rot_constant_threshold,ate_pos_constant_threshold)


# # ######################################## 2pm ##################################
# frames = [i for i in range(1,216)]
# fig, axs = plt.subplots(1, 2)
# axs[0].plot(frames, distances_2pm, label='2pm')
# axs[0].legend()
# axs[0].set(xlabel='Frame', ylabel='Position error')
# axs[1].plot(frames, angles_2pm, label='2pm')
# axs[1].legend()
# axs[1].set(xlabel='Frame', ylabel='Orientation error ($^\circ$)')
# plt.show()

# # ######################################## 6pm ##################################
# frames = [i for i in range(1,188)]
# fig, axs = plt.subplots(1, 2)
# axs[0].plot(frames, distances_6pm, label='6pm')
# axs[0].legend()
# axs[0].set(xlabel='Frame', ylabel='Position error')
# axs[1].plot(frames, angles_6pm, label='6pm')
# axs[1].legend()
# axs[1].set(xlabel='Frame', ylabel='Orientation error ($^\circ$)')
# plt.show()

# ######################################## 8pm ##################################
# frames = [i for i in range(1,168)]
# fig, axs = plt.subplots(1, 2)
# axs[0].plot(frames, distances_8pm, label='8pm')
# axs[0].legend()
# axs[0].set(xlabel='Frame', ylabel='Position error')
# axs[1].plot(frames, angles_8pm, label='8pm')
# axs[1].legend()
# axs[1].set(xlabel='Frame', ylabel='Orientation error ($^\circ$)')
# plt.show()
# # Compute translation error and rotation error 
# pass
# Plot two trajectory, black is colmap result and green is estimation results that are close to colmap and red is estimation results that are far to colmap
