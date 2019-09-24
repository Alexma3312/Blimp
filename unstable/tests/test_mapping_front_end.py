"""Unit Test to test Front End."""
import unittest
import numpy as np
import sys

sys.path.append('../') 
from sfm import mapping_front_end

# - [ ] In the future, the keypoint and descriptors size transformation pipeline should be included in the test cases.  

class TestFrontEnd(unittest.TestCase):
    def setUp(self):
        self.fe = mapping_front_end.MappingFrontEnd()
        self.img_paths = img_paths = ['datasets/4.5/mapping/1_frame.jpg', 'datasets/4.5/mapping/2.1_frame.jpg', 'datasets/4.5/mapping/2.2_frame.jpg', 'datasets/4.5/mapping/2.3_frame.jpg', 'datasets/4.5/mapping/2.4_frame.jpg', 'datasets/4.5/mapping/2.5_frame.jpg', 'datasets/4.5/mapping/2_frame.jpg', 'datasets/4.5/mapping/3_frame.jpg', 'datasets/4.5/mapping/4_frame.jpg', 'datasets/4.5/mapping/5_frame.jpg', 'datasets/4.5/mapping/6_frame.jpg']

    # def test_get_image_paths(self):
    #     self.fe.get_image_paths()
    #     self.assertEqual(self.fe.img_paths,self.img_paths)

    # def test_read_image(self):
    #     for path in self.img_paths:
    #         img = self.fe.read_image(path)
    #         assert img.any()

    # def test_superpoint_generator(self):
    #     for path in self.img_paths:
    #         img = self.fe.read_image(path)
    #         superpoints, descriptors = self.fe.superpoint_generator(img)
    #         assert len(superpoints)
    #         # print('\nThere are ',len(superpoints) ,' Superpoints:\n', superpoints)
    #         # print('Descriptors:\n', descriptors)

    # def test_extract_all_image_features(self):
    #     self.fe.get_image_paths()
    #     self.fe.extract_all_image_features()
    #     assert len(self.fe.img_pose)

    def test_feature_matching(self):
        self.fe.get_image_paths()
        self.fe.extract_all_image_features()
        self.fe.feature_matching()

    def test_read_cvs_file(self):
        kp_data = self.fe.read_kp_cvs_file('datasets/4.5/mapping/kp/0_1_keypoints.csv')
        desc_data = self.fe.read_desc_cvs_file('datasets/4.5/mapping/desc/0_1_descriptors.csv')
        assert kp_data.shape[1] == 6 
        assert desc_data.shape[1] == 512


        

if __name__ == "__main__":
    unittest.main()
