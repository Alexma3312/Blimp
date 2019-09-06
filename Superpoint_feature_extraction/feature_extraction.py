# cSpell: disable=invalid-name
import glob
import os

import cv2
import numpy as np

from SuperPointPretrainedNetwork.demo_superpoint import (PointTracker,
                                                         SuperPointFrontend)
from feature_matcher.parser import get_matches

# pylint: disable=no-member

# Font parameters for visualizaton.
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_CLR = (255, 255, 255)
FONT_PT = (4, 12)
FONT_SC = 0.4
# Jet colormap for visualization.
MYJET = np.array([[0., 0., 0.5],
                  [0., 0., 0.99910873],
                  [0., 0.37843137, 1.],
                  [0., 0.83333333, 1.],
                  [0.30044276, 1., 0.66729918],
                  [0.66729918, 1., 0.30044276],
                  [1., 0.90123457, 0.],
                  [1., 0.48002905, 0.],
                  [0.99910873, 0.07334786, 0.],
                  [0.5, 0., 0.]])


class FeatureExtraction(object):
    """Save superpoint extracted features to files."""

    def __init__(self, image_directory_path='SuperPointPretrainedNetwork/feature_extraction/undistort_images/', image_extension='*.jpg', image_size=(640, 480), nn_thresh=0.7):
        self.basedir = image_directory_path
        self.img_extension = image_extension
        self.nn_thresh = nn_thresh
        self._fe = SuperPointFrontend(weights_path="SuperPointPretrainedNetwork/superpoint_v1.pth",
                                      nms_dist=4,
                                      conf_thresh=0.015,
                                      nn_thresh=0.7,
                                      cuda=False)
        self.point_tracker = PointTracker(5, self.nn_thresh)
        self.img_size = image_size
        self.img_paths = self.get_image_paths()

    def read_image(self, impath, color=0):
        """ Read image as grayscale.
        Inputs:
            impath - Path to input image.
            img_size - (W, H) tuple specifying resize size.
        Returns:
            grayim - float32 numpy array sized H x W with values in range [0, 1].
        """
        grayim = cv2.imread(impath, color)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        # Image is resized via opencv.
        interp = cv2.INTER_AREA
        grayim = cv2.resize(
            grayim, (self.img_size[0], self.img_size[1]), interpolation=interp)
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

        superpoints, descriptors, heatmap = self._fe.run(image)

        return superpoints[:2, ].T, descriptors.T, heatmap

    def get_image_paths(self):
        """Get all image paths within the directory."""
        print('==> Processing Image Directory Input.')
        search = os.path.join(self.basedir, self.img_extension)
        img_paths = glob.glob(search)
        img_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        print("Number of Images: ", len(img_paths))
        maxlen = len(img_paths)
        if maxlen == 0:
            raise IOError(
                'No images were found (maybe wrong \'image extension\' parameter?)')
        return img_paths

    def extract_all_image_features(self):
        """Extract features for each image within the image path list"""
        for i, impath in enumerate(self.img_paths):
            grayim = self.read_image(impath)
            keypoints, descriptors, heatmap = self.superpoint_generator(grayim)
            self.draw_features(keypoints, grayim, heatmap, i)
            self.save_to_file(keypoints, descriptors, i)

    def save_to_file(self, kp_data, desc_data, index):
        """This is used to save the features information of each frame into a .key file,[x,y, desc(256)]"""
        nrpoints = kp_data.shape[0]
        descriptor_length = 256

        features = [np.hstack((point, desc_data[i]))
                    for i, point in enumerate(kp_data)]
        features = np.array(features)

        dir_name = self.basedir+'features/'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        # np.savetxt(dir_name+self.leading_zero(index) +
        #            '.key', features, fmt='%.4f')
        np.savetxt(dir_name+self.leading_zero(index) +
                   '.key', features)

        first_line = str(nrpoints)+' '+str(descriptor_length)+'\n'
        with open(dir_name+self.leading_zero(index)+'.key', 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(first_line+content)

    def leading_zero(self, index):
        """Create leading zero filename"""
        index_string = str(index)
        index_len = len(index_string)
        output = ['0' for i in range(7)]
        for i in range(index_len):
            output[7-index_len+i] = index_string[i]
        return ''.join(output)

    def draw_features(self, keypoints, img, heatmap, index):
        """Draw feature images and heatmap images."""
        # Extra output -- Show current point detections.
        out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        for pt in keypoints:
            pt1 = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(out1, pt1, 1, (0, 255, 0), -1, lineType=16)
            cv2.putText(out1, 'Raw Point Detections', FONT_PT,
                        FONT, FONT_SC, FONT_CLR, lineType=16)

        # Extra output -- Show the point confidence heatmap.
        if heatmap is not None:
            min_conf = 0.001
            heatmap[heatmap < min_conf] = min_conf
            heatmap = -np.log(heatmap)
            heatmap = (heatmap - heatmap.min()) / \
                (heatmap.max() - heatmap.min() + .00001)
            out2 = MYJET[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
            out2 = (out2*255).astype('uint8')
        else:
            out2 = np.zeros_like(out1)
            cv2.putText(out2, 'Raw Point Confidences', FONT_PT,
                        FONT, FONT_SC, FONT_CLR, lineType=16)

        out_dir_1 = self.basedir+'feature_image/'
        if not os.path.exists(out_dir_1):
            os.mkdir(out_dir_1)
        out_file_1 = out_dir_1+'frame_%05d' % index+'.jpg'
        print('Writing image to %s' % out_file_1)
        cv2.imwrite(out_file_1, out1)

        out_dir_2 = self.basedir+'heatmap/'
        if not os.path.exists(out_dir_2):
            os.mkdir(out_dir_2)
        out_file_2 = out_dir_2+'heatmap_%05d' % index+'.jpg'
        print('Writing image to %s' % out_file_2)
        cv2.imwrite(out_file_2, out2)

    def get_all_feature_matches(self, calibration, threshold=1):
        """Get feature matches of every two images"""
        image_number = len(self.img_paths)
        for i in range(image_number-1):
            for j in range(i+1, image_number):
                grayim_1 = self.read_image(self.img_paths[i])
                grayim_2 = self.read_image(self.img_paths[j])
                keypoints_1, descriptors_1, _ = self.superpoint_generator(
                    grayim_1)
                keypoints_2, descriptors_2, _ = self.superpoint_generator(
                    grayim_2)
                matches = self.point_tracker.nn_match_two_way(
                    descriptors_1.T, descriptors_2.T, self.nn_thresh)
                bad_essential_matrix, good_matches = self.ransac_filter_opencv(
                    matches, keypoints_1, keypoints_2, threshold, calibration)
                if bad_essential_matrix:
                    print(
                        "Not enough points to generate essential matrix for image_", i, " and image_", j)
                    continue
                print("Matches between image {} and image {} (ransac filter/two way NN matches): ".format(i, j),
                      good_matches.shape[0], "/", matches.shape[1])
                self.save_feature_matches(i, j, good_matches)
                self.save_match_images(
                    i, j, good_matches, keypoints_1, keypoints_2)

    def ransac_filter_opencv(self, matches, keypoints_1, keypoints_2, threshold, calibration):
        """Use opencv ransac to filter matches."""
        src = np.array([], dtype=np.float).reshape(0, 2)
        dst = np.array([], dtype=np.float).reshape(0, 2)
        for match in matches.T:
            src = np.vstack((src, keypoints_1[int(match[0])]))
            dst = np.vstack((dst, keypoints_2[int(match[1])]))

        if src.shape[0] < 20:
            return True, np.array([])

        src = np.expand_dims(src, axis=1)
        dst = np.expand_dims(dst, axis=1)
        E, mask = cv2.findEssentialMat(
            dst, src, cameraMatrix=calibration, method=cv2.RANSAC, prob=0.999, threshold=threshold)
        print("E:\n",E)
        R1, R2, T = cv2.decomposeEssentialMat(E)
        print("R1:\n",R1)
        print("R2:\n",R2)
        print("T:\n",T)
        fundamental_mat, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 1, 0.99)
        print("fundamental_mat:\n",fundamental_mat)

        if mask is None:
            return True, np.array([])
        good_matches = [matches.T[i]
                        for i, score in enumerate(mask) if score == 1]
        good_matches = np.array(good_matches)[:, :2]

        return False, good_matches

    def save_feature_matches(self, idx1, idx2, matches, save_dir='matches/'):
        """Save the feature matches of index 1 image and index 2 image."""
        matches_number = matches.shape[0]
        idx1_col = np.array([idx1]*matches_number).reshape(matches_number, 1)
        idx2_col = np.array([idx2]*matches_number).reshape(matches_number, 1)
        match_result = np.concatenate((idx1_col, matches[:, 0].reshape(
            matches_number, 1), idx2_col, matches[:, 1].reshape(matches_number, 1)), axis=1)
        dir_name = self.basedir+save_dir
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        file_name = dir_name+'match_{}_{}.dat'.format(idx1, idx2)
        np.savetxt(file_name,
                   match_result, fmt='%d')
        with open(file_name, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write("/* Format: \n frame_1_idx frame_2_idx \n num_of_matches\n frame_1_idx feature_idx frame_2_idx feature_idx */\n" +
                    str(idx1)+' '+str(idx2)+'\n'+str(matches_number)+'\n'+content)

    def save_match_images(self, idx1, idx2, matches, keypoints_1, keypoints_2):
        """Create an image to display matches between an image pair."""
        dir_name = self.basedir+'match_images/'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        file_name = dir_name+'match_{}_{}.jpg'.format(idx1, idx2)
        img1 = cv2.imread(self.img_paths[idx1])
        img2 = cv2.imread(self.img_paths[idx2])
        vis = np.concatenate((img1, img2), axis=1)

        for match in matches:
            pt_src = (int(keypoints_1[int(match[0])][0]), int(
                keypoints_1[int(match[0])][1]))
            pt_dst = (int(keypoints_2[int(match[1])][0]+640),
                      int(keypoints_2[int(match[1])][1]))

            cv2.circle(vis, pt_src, 3, (0, 255, 0), -1, lineType=16)
            cv2.circle(vis, pt_dst, 3, (0, 255, 0), -1, lineType=16)
            cv2.line(vis, pt_src, pt_dst, (255, 0, 255), 1)
        cv2.imwrite(file_name, vis)

    def filter_match_with_two_way_nn(self):
        """Performs two-way nearest neighbor matching of two sets of descriptors, 
            such that the NN match from descriptor A->B must equal the NN match from B->A"""
        dir_name_1 = self.basedir+'matches/'
        dir_name_2 = self.basedir+'4d_matches/'
        image_number = len(self.img_paths)
        for i in range(image_number-1):
            for j in range(i+1, image_number):
                file_name_1 = dir_name_1+'match_{}_{}.dat'.format(i, j)
                _, matches = get_matches(file_name_1)
                matches = np.array(matches)[:, (1, 3)]

                file_name_2 = dir_name_2+'match_{}_{}.dat'.format(i, j)
                _, matches_4d = get_matches(file_name_2)
                matches_4d = np.array(matches_4d)[:, (1, 3)]

                # Get intersecting rows across two 2D numpy arrays
                new_matches = np.array(
                    [x for x in set(tuple(x) for x in matches) & set(tuple(x) for x in matches_4d)])
                print("matches:", matches.shape, "matches_4d:",
                      matches_4d.shape, 'new_matches:', new_matches.shape)
                self.save_feature_matches(i, j, new_matches, "new_matches/")
