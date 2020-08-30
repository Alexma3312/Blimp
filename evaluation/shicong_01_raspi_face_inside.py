from evaluation.read_colmap_pose import read_colmap_pose_result
from evaluation.read_pose_estimation_result import read_pose_estimation_result
from sfm.sim3 import Similarity3
from evaluation.utils import difference
import matplotlib.pyplot as plt


# Read colmap pose
colmap_image_text_file = "/home/sma96/Blimps/evaluation/face_inside/groundtruth/images.txt"
# there are 534 images, 0~136 are mapping images, 137~523 are ground truth poses
colmap_poses, colmap_bad_indices = read_colmap_pose_result(colmap_image_text_file, 524, 137, 523)

# Read pose estimation result
poses_dat_file = "/home/sma96/Blimps/evaluation/face_inside/poses.dat"
estimated_poses, estimate_bad_indices = read_pose_estimation_result(poses_dat_file, 387, 5)

# sim3 transform
# Read colmap map poses
size = 137
colmap_image_text_file = "/home/sma96/Blimps/evaluation/face_inside/groundtruth/images.txt"
groundtruth_map_poses, groundtruth_map_bad_indices = read_colmap_pose_result(colmap_image_text_file, size, 0, size-1)
colmap_image_text_file = "/home/sma96/Blimps/evaluation/face_inside/map/images.txt"
map_poses, map_bad_indices = read_colmap_pose_result(colmap_image_text_file, size, 0, size-1)

# Generate pose pairs
bad_indices = groundtruth_map_bad_indices | map_bad_indices
pose_pairs = []
for i in range(size):
    if i not in bad_indices:
        pose_pairs.append([map_poses[i],groundtruth_map_poses[i]])
# Get sim3 transform parameter
sim3 = Similarity3()
sim3.sim3_pose(pose_pairs)

# Transform estimated poses to groundtruth poses
bad_indices = colmap_bad_indices | estimate_bad_indices
distances = []
angles = []
for i in range(387):
    if i not in bad_indices:
        estimated_pose = sim3.pose(estimated_poses[i])
        distance, angle = difference(estimated_pose, colmap_poses[i])
        distances.append(distance)
        angles.append(angle)
        pass


frames = [i for i in range(1,388)]
plt.plot(frames, distances)
plt.xlabel('Frame')
plt.ylabel('Distance error')
plt.show()
plt.plot(frames, angles)
plt.xlabel('Frame')
plt.ylabel('Angle error ($^\circ$)')
plt.show()
# Compute translation error and rotation error 
pass
# Plot two trajectory, black is colmap result and green is estimation results that are close to colmap and red is estimation results that are far to colmap
 