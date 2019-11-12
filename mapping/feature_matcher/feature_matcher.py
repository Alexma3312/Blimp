"""Feature Matcher"""
import os

# cSpell: disable=invalid-name
# pylint: disable=no-member
from calibration.undistort_images.undistort_images import undistort
from calibration.undistort_images.undistort_images_reserve_source_points import undistort as undistort_shrink
from superpoint_descriptor.superpoint_wrapper import SuperpointWrapper


class FeatureMatcher():
    """Feature Matcher"""

    def __init__(self, basedir, img_extension, img_size, number_images):
        """
        Feature Matcher.
        """
        self.height, self.width = img_size
        self.image_extension = img_extension
        self.basedir = basedir
        self.number_images = number_images

        self.source_directory = self.basedir+'source_images/'
        self.undistort_dir = self.basedir+'undistort_images/'

    def undistortion(self, distort_calib, distort_coeff, resize_output, output_prefix="frame"):
        """Undistort Images."""
        assert os.path.exists(
            self.source_directory), "Source Directory Dont Exist. Please move all source images into a directory named 'source_images' under the basedir."
        if resize_output is True:
            undistort(self.source_directory, self.image_extension, self.undistort_dir, output_prefix,
                      distort_calib, distort_coeff)
        else:
            undistort_shrink(self.source_directory, self.image_extension, self.undistort_dir, output_prefix,
                             distort_calib, distort_coeff)

    def feature_extraction(self, new_image_size, desc_type, nn_thresh=0.7):
        """Extract features."""
        if desc_type == 'Superpoint':
            self.desc_size = 256
            superpoint_wrapper = SuperpointWrapper(
                self.basedir, self.image_extension, new_image_size, nn_thresh)
            # Extract and Save feature information into files
            superpoint_wrapper.extract_all_image_features()

        if desc_type == 'ORB':
            self.desc_size = 128

    def feature_matching(self, new_image_size, desc_type, matching_type, calibration, threshold=1, nn_thresh=0.7):
        """Match features."""
        if desc_type == 'Superpoint':
            superpoint_wrapper = SuperpointWrapper(
                self.basedir, self.image_extension, new_image_size, nn_thresh)

            if matching_type == 'Two Way NN':
                # Extract and Save feature information into files
                # superpoint_wrapper.get_all_feature_matches(calibration, threshold)
                superpoint_wrapper.get_all_feature_matches(calibration)

            if matching_type == 'FLANN':
                # Extract and Save feature information into files
                superpoint_wrapper.get_all_feature_matches_FLANN(calibration)

            if matching_type == 'Select FLANN':
                # Extract and Save feature information into files
                superpoint_wrapper.get_select_feature_matches_FLANN(
                    calibration)

            if matching_type == 'FLANN SAVE':
                # Extract and Save feature information into files
                superpoint_wrapper.robust_feature_matches(calibration)
