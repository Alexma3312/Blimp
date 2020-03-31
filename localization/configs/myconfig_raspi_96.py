"""Configuration File"""
import gtsam
from localization.camera import Camera
from gtsam import Cal3_S2
import numpy as np


measurement_noise_sigma = 1.0
measurement_noise = gtsam.noiseModel_Robust(gtsam.noiseModel_mEstimator_Huber(
    1.345), gtsam.noiseModel_Isotropic.Sigma(2, measurement_noise_sigma))

# Because the map is known, we use the landmarks from the visible map with nearly zero error as priors.
point_prior_noise = gtsam.noiseModel_Isotropic.Sigma(3, 0.001)
pose_translation_prior_noise = gtsam.noiseModel_Isotropic.Sigma(3, 0.1)
noise_models = [measurement_noise,
                point_prior_noise, pose_translation_prior_noise]


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
calibration_matrix = Cal3_S2(fx=fx, fy=fy, s=s,
                             u0=u0, v0=v0)
distortion = np.array([k1, k2, p1, p2, p3])
image_size = (int(640), int(480))
camera = Camera(calibration_matrix, distortion, image_size)
