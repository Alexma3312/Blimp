
import matplotlib.pyplot as plt

from evaluation.utils import get_distance_angle_error_illumination, ate



poses_dat_file =  "/home/sma96/Blimps/evaluation/illumination/result/2pm/poses.dat"
ground_truth_image_text_file =  "/home/sma96/Blimps/evaluation/illumination/ground_truth/2pm/images.txt"
colmap_image_text_file =  "/home/sma96/Blimps/evaluation/illumination/map/images.txt"
number_poses = 215
sample_rate = 5
map_number_poses = 136
distances_2pm, angles_2pm = get_distance_angle_error_illumination(poses_dat_file, ground_truth_image_text_file, colmap_image_text_file, number_poses, sample_rate, map_number_poses)
distances_2pm =[distance*3.1 for distance in distances_2pm]
ate_rot_2pm = ate(angles_2pm)
ate_pos_2pm = ate(distances_2pm)
print(ate_rot_2pm,ate_pos_2pm)


poses_dat_file = "/home/sma96/Blimps/evaluation/illumination/result/6pm/poses.dat"
ground_truth_image_text_file =  "/home/sma96/Blimps/evaluation/illumination/ground_truth/6pm/images.txt"
colmap_image_text_file =  "/home/sma96/Blimps/evaluation/illumination/map/images.txt"
number_poses = 187
sample_rate = 5
map_number_poses = 136
distances_6pm, angles_6pm = get_distance_angle_error_illumination(poses_dat_file, ground_truth_image_text_file, colmap_image_text_file, number_poses, sample_rate, map_number_poses)
distances_6pm =[distance*3.1 for distance in distances_6pm]
ate_rot_6pm = ate(angles_6pm)
ate_pos_6pm = ate(distances_6pm)
print(ate_rot_6pm,ate_pos_6pm)

poses_dat_file = "/home/sma96/Blimps/evaluation/illumination/result/8pm/poses.dat"
ground_truth_image_text_file =  "/home/sma96/Blimps/evaluation/illumination/ground_truth/8pm/images.txt"
colmap_image_text_file =  "/home/sma96/Blimps/evaluation/illumination/map/images.txt"
number_poses = 167
sample_rate = 5
map_number_poses = 136
distances_8pm, angles_8pm = get_distance_angle_error_illumination(poses_dat_file, ground_truth_image_text_file, colmap_image_text_file, number_poses, sample_rate, map_number_poses)
distances_8pm =[distance*3.1 for distance in distances_8pm]
ate_rot_8pm = ate(angles_8pm)
ate_pos_8pm = ate(distances_8pm)
print(ate_rot_8pm,ate_pos_8pm)

# ######################################## 2pm ##################################
frames = [i for i in range(1,216)]
fig, axs = plt.subplots(1, 2)
axs[0].plot(frames, distances_2pm, label='2pm')
axs[0].legend()
axs[0].set(xlabel='Frame', ylabel='Position error')
axs[1].plot(frames, angles_2pm, label='2pm')
axs[1].legend()
axs[1].set(xlabel='Frame', ylabel='Orientation error ($^\circ$)')
plt.show()

# ######################################## 6pm ##################################
frames = [i for i in range(1,188)]
fig, axs = plt.subplots(1, 2)
axs[0].plot(frames, distances_6pm, label='6pm')
axs[0].legend()
axs[0].set(xlabel='Frame', ylabel='Position error')
axs[1].plot(frames, angles_6pm, label='6pm')
axs[1].legend()
axs[1].set(xlabel='Frame', ylabel='Orientation error ($^\circ$)')
plt.show()

######################################## 8pm ##################################
frames = [i for i in range(1,168)]
fig, axs = plt.subplots(1, 2)
axs[0].plot(frames, distances_8pm, label='8pm')
axs[0].legend()
axs[0].set(xlabel='Frame', ylabel='Position error')
axs[1].plot(frames, angles_8pm, label='8pm')
axs[1].legend()
axs[1].set(xlabel='Frame', ylabel='Orientation error ($^\circ$)')
plt.show()
# Compute translation error and rotation error 
pass
# Plot two trajectory, black is colmap result and green is estimation results that are close to colmap and red is estimation results that are far to colmap
