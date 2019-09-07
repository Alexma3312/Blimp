"""An example script to extract and save Superpoint feature, Superpoint feature matches and images."""
# cSpell: disable=invalid-name
# pylint: disable=no-member,no-name-in-module,wrong-import-order
from superpoint_descriptor.superpoint_wrapper import SuperpointWrapper
from gtsam import Cal3_S2


def run_feature_extraction():
    """Feature Extraction."""
    image_directory_path = 'superpoint_descriptor/undistort_images/'
    image_extension = '*.jpg'
    image_size = (640, 480)
    nn_thresh = 0.7
    superpoint_wrapper = SuperpointWrapper(
        image_directory_path, image_extension, image_size, nn_thresh)
    # Extract and Save feature information into files
    superpoint_wrapper.extract_all_image_features()


def run_matching(superpoint_wrapper):
    """Create Feature Matching after Feature Extraction."""
    # Create matches and save both the information and the images(with matches displayed on the origin images)
    calibration = Cal3_S2(fx=232.0542, fy=252.8620, s=0,
                          u0=325.3452, v0=240.2912).matrix()
    superpoint_wrapper.get_all_feature_matches(calibration, 1)


def run_filter_match_with_two_way_nn(superpoint_wrapper):
    """Filter 4d agri matches with two way nn matches (normally not used)"""
    superpoint_wrapper.filter_match_with_two_way_nn()


if __name__ == "__main__":
    run_feature_extraction()
