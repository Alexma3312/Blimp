"""Configuration File"""
import gtsam
from localization.camera import Camera
from gtsam import Cal3_S2
import numpy as np
import math


measurement_noise_sigma = 5.0
measurement_noise = gtsam.noiseModel_Robust(gtsam.noiseModel_mEstimator_Huber(
    1.345), gtsam.noiseModel_Isotropic.Sigma(2, measurement_noise_sigma))
# Because the map is known, we use the landmarks from the visible map with nearly zero error as priors.
point_prior_noise = gtsam.noiseModel_Isotropic.Sigma(3, 0.01)
degree = 5
angle = math.radians(degree)
pose_prior_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array(
                [0.1, 0.1, 0.1, angle, angle, angle]))  # 10cm std on x,y,z 5 degree on roll,pitch,yaw
noise_models = [measurement_noise,
                point_prior_noise, pose_prior_noise]


# Create calibration matrix
fx = 501.582483
fy = 505.573729
s = 0
u0 = 312.511059
v0 = 241.207213
k1 = 0.167262
k2 = -0.260489
p1 = -0.000013
p2 = -0.002028
p3 = 0.000000
calibration_matrix = Cal3_S2(fx=fx/2, fy=fy/2, s=s,
                             u0=u0/2, v0=v0/2)
distortion = np.array([k1, k2, p1, p2, p3])
image_size = (int(640/2), int(480/2))
camera = Camera(calibration_matrix, distortion, image_size)
