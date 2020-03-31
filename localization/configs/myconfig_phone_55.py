import gtsam
from localization.camera import Camera


measurement_noise_sigma = 1.0
measurement_noise = gtsam.noiseModel_Robust(gtsam.noiseModel_mEstimator_Huber(
            1.345), gtsam.noiseModel_Isotropic.Sigma(2, measurement_noise_sigma))

# Because the map is known, we use the landmarks from the visible map with nearly zero error as priors.
point_prior_noise = gtsam.noiseModel_Isotropic.Sigma(3, 0.001)
pose_translation_prior_noise = gtsam.noiseModel_Isotropic.Sigma(3, 0.1)
noise_models = [measurement_noise, point_prior_noise,pose_translation_prior_noise]


# Create calibration matrix
# fx = 1383.203176
# fy = 1384.255167
# u0 = 923.818559
# v0 = 533.145671
fx = 1397.433127
fy = 1398.897073 
u0 = 963.919974 
v0 = 517.067033 
# 0.008088, -0.011712, -0.005013, 0.003599
image_size = (1920, 1080)
camera = Camera(fx, fy, u0, v0, image_size)
