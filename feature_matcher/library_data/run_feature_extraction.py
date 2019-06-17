# cSpell: disable=invalid-name
from feature_extraction.feature_extraction import FeatureExtraction

# pylint: disable=no-member


def run():
    """Execution"""
    image_directory_path = 'feature_matcher/library_data/dark/undistort_images/'
    image_extension = '*.jpg'
    image_size = (640, 480)
    nn_thresh = 0.7
    feature_extract = FeatureExtraction(
        image_directory_path, image_extension, image_size, nn_thresh)
    feature_extract.get_image_paths()
    feature_extract.extract_all_image_features()


if __name__ == "__main__":
    run()
