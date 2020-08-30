"""Configuration File"""
import gtsam
from localization.camera import Camera
from gtsam import Cal3_S2
import numpy as np


measurement_noise_sigma = 5.0
measurement_noise = gtsam.noiseModel_Robust(gtsam.noiseModel_mEstimator_Huber(
    1.345), gtsam.noiseModel_Isotropic.Sigma(2, measurement_noise_sigma))

# Because the map is known, we use the landmarks from the visible map with nearly zero error as priors.
point_prior_noise = gtsam.noiseModel_Isotropic.Sigma(3, 0.01)
pose_translation_prior_noise = gtsam.noiseModel_Isotropic.Sigma(3, 0.1)
noise_models = [measurement_noise,
                point_prior_noise, pose_translation_prior_noise]


# Create calibration matrix
fx = 458.654
fy = 457.296
s = 0
u0 = 367.215
v0 = 248.375
k1 = -0.28340811
k2 = 0.07395907
p1 = 0.00019359
p2 = 1.76187114e-05
p3 = 0.000000
calibration_matrix = Cal3_S2(fx=fx, fy=fy, s=s,
                             u0=u0, v0=v0)
distortion = np.array([k1, k2, p1, p2, p3])
image_size = (int(752), int(480))
camera = Camera(calibration_matrix, distortion, image_size)
