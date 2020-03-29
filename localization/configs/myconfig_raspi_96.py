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
downsample_scale = 4
fx = 501.582483/downsample_scale
fy = 505.573729/downsample_scale
u0 = 312.511059/downsample_scale
v0 = 241.207213/downsample_scale
image_size = (int(640/downsample_scale), int(480/downsample_scale))
camera = Camera(fx, fy, u0, v0, image_size)
