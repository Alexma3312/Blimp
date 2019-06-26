"""This is a mapping front end to preprocess input images."""
import cv2
import sys
import glob
import numpy as np
import gtsam
from gtsam import Point3, Rot3, Pose3
from SuperPointPretrainedNetwork.demo_superpoint import *

sys.path.append('../')
# https://gist.github.com/abidrahmank/6450018

class ImagePose(object):
    def __init__(self):
        self.desc = np.array([])
        self.R = Rot3()
        self.t = Point3()
        self.T = Pose3()
        self.P = np.zeros((3,4))
        self.kp_matches = {}
        self.kp_landmark = {}
    
    def kp_match_idx(self, kp_idx, img_idx):
        # use current frame key point idx and an image index to return the matched key point in the img_idx image
        return self.kp_matches['kp_idx']['img_idx']

    def kp_match_exist(self, kp_idx, img_idx):
        # check if there are matched keypoints in the img index
        return True if 'img_idx' in self.kp_matches['kp_idx'] else False

    def kp_3d(self, kp_idx):
        # use current frame key point idx to find the corresponding landmark point
        return self.kp_landmark['kp_idx']

    def kp_3d_exist(self, kp_idx):
        # check if there are corresponding landmark
        return True if 'kp_idx' in self.kp_landmark else False

class LandMark(object):
    def __init__(self,point=Point3(),seen = 0):
        self.point = point
        self.seen = seen

class SFMHelper(object):
    def __init__(self, img_pose=[],landmark=[]):
        self.img_pose=img_pose
        self.landmark=landmark

class MappingFrontEnd(object):
    def __init__(self,image_directory_path = '~/datasets/4.5/mapping/', image_extension = '*.jpg', image_size = (160,120), down_sample = 4, nn_thresh = 0.7):
        self.basedir = image_directory_path
        self.img_extension = image_extension
        self.nn_thresh = nn_thresh
        self.fe = SuperPointFrontend(weights_path="SuperPointPretrainedNetwork/superpoint_v1.pth",
                                nms_dist=4,
                                conf_thresh=0.015,
                                nn_thresh=0.7,
                                cuda=False)
        self.img_size = image_size
        self.scale = down_sample

        self.keypoint_list = []
        self.descriptor_list = []
        self.correspondence_table = []
        
        self.sfm_helper = SFMHelper()

    def read_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs:
            impath - Path to input image.
            img_size - (W, H) tuple specifying resize size.
        Returns:
            grayim - float32 numpy array sized H x W with values in range [0, 1].
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        # Image is resized via opencv.
        interp = cv2.INTER_AREA
        grayim = cv2.resize(
            grayim, (self.img_size[1], self.img_size[0]), interpolation=interp)
        grayim = (grayim.astype('float32') / 255.)
        return grayim

    def superpoint_generator(self, image):
        """Use superpoint to extract features in the image
        Returns:
            superpoint - Nx2 (gtsam.Point2) numpy array of 2D point observations.
            descriptors - Nx256 numpy array of corresponding unit normalized descriptors.

        Refer to /SuperPointPretrainedNetwork/demo_superpoint for more information about the parameters
        Output of SuperpointFrontend.run():
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
        """
        
        superpoints, descriptors, _ = self.fe.run(image)

        # Transform superpoints from 3*N numpy array to N*2 numpy array
        superpoints = np.transpose(superpoints[:2, ])

        # Transform descriptors from 256*N numpy array to N*256 numpy array
        # descriptors = np.transpose(descriptors)

        # Transform superpoint into gtsam.Point2 and store in a N*2 list
        superpoints_reformat = [Point2(self.scale*point[0],self.scale*point[1]) for point in superpoints]

        return superpoints_reformat, descriptors

    def get_image_paths(self):
        """Get all image paths within the directory."""
        print('==> Processing Image Directory Input.')
        search = os.path.join(self.basedir, self.img_extension)
        self.img_paths = glob.glob(search)
        self.img_paths.sort()
        maxlen = len(self.img_paths)
        if maxlen == 0:
          raise IOError('No images were found (maybe wrong \'image extension\' parameter?)')
    
    def extract_all_image_features(self):
        """"""
        for impath in self.img_paths:
            grayim = self.read_image(impath)
            keypoints, descriptors = self.superpoint_generator(grayim)
            self.keypoint_list.append(keypoints)
            self.descriptor_list.append(descriptors)

    def generate_cvs_file(self):
        """Output convention [source_image_keypoints(u,v), destination_image_keypoints(u,v)]"""
        # This is used to save the features information of each frame into a .txt file,
        np.savetxt('dataset/key_points/key_points_%05d.txt' % vs.i,np.transpose(pts[:2,]), fmt='%d')
        # np.savetxt('dataset/features/features_%05d.txt' % vs.i,np.transpose(desc), fmt='%f')
        np.savetxt('dataset/features/features_%05d.csv'% vs.i,np.transpose(desc),fmt='%10.5f',delimiter=',')

    def feature_matching(self):
        """
        
        Feature will be matched based on:
            - L2 distances of descriptors
            - findFundamentalMat
        """
        kp_len = len(self.keypoint_list)
        s_matched_idx = []
        d_matched_idx = []
        mask = []

        # Iterate through all images.
        for i in range(0,kp_len-1):
            for j in range(i+1,kp_len):
                matches = PointTracker.nn_match_two_way(self.descriptor_list[i], self.descriptor_list[j], self.nn_thresh)
                cv2.findFundamentalMat(matches[0],matches[1])
                
                good_match_count = 0
                for good_match in mask:
                    if good_match == 1:
                        good_match_count+=1

            print("Feature matching ",i," ",j, " ==> ", good_match_count, "/" <<len(mask))

                # Iterate through all keypoints.

    def recover_poses(self):
        return

    def recover_map_points(self):
        return