import gtsam
from localization.camera import Camera


measurement_noise_sigma = 1.0
measurement_noise = gtsam.noiseModel_Isotropic.Sigma(
    2, measurement_noise_sigma)

# Because the map is known, we use the landmarks from the visible map with nearly zero error as priors.
point_prior_noise = gtsam.noiseModel_Isotropic.Sigma(3, 0.01)


# Create calibration matrix
fx = 1406.5/3
fy = 1317/3
u0 = 775.2312/3
v0 = 953.3863/3
image_size = (640, 480)
camera = Camera(fx, fy, u0, v0, image_size)
