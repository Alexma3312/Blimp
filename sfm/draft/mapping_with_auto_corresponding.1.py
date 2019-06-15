"""This is a mapping front end to preprocess input images."""
import cv2
import sys
import glob
import numpy as np
import gtsam
from gtsam import Point2,Point3, Rot3, Pose3, symbol
from SuperPointPretrainedNetwork.demo_superpoint import *
import math
from datasets.localization import sim3_transform_map,plot_localization
from atrium_control.feature import Features
from atrium_control.map import Map
import time
from sfm import sim3
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import statistics as st
from scipy import stats


def X(i):
    """Create key for pose i."""
    return symbol(ord('x'), i)


def P(j):
    """Create key for landmark j."""
    return symbol(ord('p'), j)


def transform_from(T, pose):
    """ Calculate the Euclidean transform of a Pose3
    Parameters:
        pose - Pose3 object
    Returns:
        Pose3 object
    """

    R = T.rotation().compose(pose.rotation())
    translation = T.rotation().rotate(pose.translation()).vector() + T.translation().vector()
    return Pose3(R, Point3(translation))

def undistort_image(basedir,img_extension):
    """A function to undistort the distorted images."""
    search = os.path.join(basedir, img_extension)
    img_paths = glob.glob(search)
    img_paths.sort()
    print("Number of Images: ",len(img_paths))
    maxlen = len(img_paths)
    if maxlen == 0:
        raise IOError('No images were found (maybe wrong \'image extension\' parameter?)')

    calibration = gtsam.Cal3_S2(fx=347.820593, fy=329.096945,s=0,u0=295.717950, v0=222.964889).matrix()
    distortion = np.array([-0.284322, 0.055723, 0.006772, 0.005264, 0.000000])
    rectification = np.identity(3)
    projection = np.array([[240.446564 ,0.000000, 302.423680, 0.000000],[0.000000, 265.140778, 221.096494, 0.000000],[0.000000, 0.000000, 1.000000, 0.000000]]) 
    
    count = 0
    img_idx = 0
    for i,img_path in enumerate(img_paths):
        img = cv2.imread(img_path, 1)
        if count == 8:
            count = 0
            img_idx += 1

        dst = cv2.undistort(img, calibration, distortion, np.eye(3), projection)

        output_path = '%d_'%img_idx+'%d_frame_undist.jpg'%count
        output_path = os.path.join(basedir+'source/', output_path)
        cv2.imwrite(output_path,dst)
        count += 1

class ImagePose(object):
    """An object to store the key point and feature descriptor information of a certain pose."""
    def __init__(self,calibration):
        self.desc = np.array([])
        self.kp = np.array([])
        self.R = Rot3()
        self.t = Point3()
   
        self.T = Pose3(self.R,self.t)
        self.P = np.dot(calibration,np.hstack((np.identity(3),np.zeros((3,1)))))
        self.kp_matches = {}
        self.kp_landmark = {}
        self.seen=0
    
    def kp_match_idx(self, kp_idx, img_idx):
        # use self.img_pose[i+1]rent frame key point idx and an image index to return the matched key point in the img_idx image
        return self.kp_matches.get(kp_idx).get(img_idx)

    def kp_match_exist(self, kp_idx, img_idx):
        # check if there are matched keypoints in the img index
        # return True if img_idx in self.kp_matches[kp_idx] else False
        if self.kp_matches.get(kp_idx) is None:
            return False
        else:
            # return self.kp_matches.get(kp_idx).get(img_idx) is not None
            if self.kp_matches.get(kp_idx).get(img_idx) == None:
                return False
            else:
                return True

    def kp_3d(self, kp_idx):
        # use self.img_pose[i+1]rent frame key point idx to find the corresponding landmark point
        return self.kp_landmark.get(kp_idx)

    def kp_3d_exist(self, kp_idx):
        # check if there are corresponding landmark
        # return self.kp_landmark.get(kp_idx) is not None
        if self.kp_landmark.get(kp_idx) == None:
            return False
        else:
            return True 

class LandMark(object):
    """An object to store the landmark point and its descriptor information."""
    def __init__(self,point=np.array([0,0,0]),seen = 0):
        self.point = point
        self.seen = seen
        self.desc = np.array([])

class MappingFrontEnd(object):
    """Include feature extraction, auto corresponding, and bundle adjustment."""
    def __init__(self,image_directory_path = 'datasets/5x5x8/source_1x5x1/', image_extension = '*.jpg', image_size = (640,480), down_sample = 1, nn_thresh = 0.7, row = 1, col = 6, angles = 1, directions = 1):
    # def __init__(self,image_directory_path = 'datasets/5x5x8/source/', image_extension = '*.jpg', image_size = (640,480), down_sample = 1, nn_thresh = 0.7, row = 5, col = 5, angles = 8, directions = 3):
    # def __init__(self,image_directory_path = 'datasets/4.5/mapping/source_images/', image_extension = '*.jpg', image_size = (640,480), down_sample = 1, nn_thresh = 0.7, row = 1, col = 6, angles = 1, directions = 1):
        self.basedir = image_directory_path
        self.img_extension = image_extension
        self.nn_thresh = nn_thresh
        self.fe = SuperPointFrontend(weights_path="SuperPointPretrainedNetwork/superpoint_v1.pth",
                                nms_dist=4,
                                conf_thresh=0.015,
                                nn_thresh=0.7,
                                cuda=False)
        self.point_tracker = PointTracker(5, self.nn_thresh)
        self.img_size = image_size
        self.scale = down_sample

        self.cal= gtsam.Cal3_S2(fx=232.0542, fy=252.8620,s=0,u0=325.3452, v0=240.2912)
        self.calibration = self.cal.matrix()

        self.keypoint_list = []
        self.descriptor_list = []
        self.correspondence_table = []
        
        self.row = row
        self.col = col
        self.angles = angles
        self.directions = directions
        self.img_pose = np.array([])
        self.landmark=[]
        self.landmark_descriptor = []

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

        superpoints, descriptors, heatmap = self.fe.run(image)
        
        # Transform superpoints from 3*N numpy array to N*2 numpy array
        superpoints = self.scale*superpoints[:2, ].T

        # Transform descriptors from 256*N numpy array to N*256 numpy array
        # descriptors = np.transpose(descriptors)

        # Transform superpoint into gtsam.Point2 and store in a N*2 list
        # superpoints_reformat = [Point2(self.scale*point[0],self.scale*point[1]) for point in superpoints]

        return superpoints, descriptors

    def get_image_paths(self):
        """Get all image paths within the directory."""
        print('==> Processing Image Directory Input.')
        search = os.path.join(self.basedir, self.img_extension)
        self.img_paths = glob.glob(search)
        self.img_paths.sort()
        print("Number of Images: ",len(self.img_paths))
        maxlen = len(self.img_paths)
        if maxlen == 0:
          raise IOError('No images were found (maybe wrong \'image extension\' parameter?)')
    
    def extract_all_image_features(self):
        """Extract features of all images in the image path."""
        row_idx = 0
        col_idx = 0
        angle_idx = 0
        angle_poses = np.array([])
        col_poses = np.array([]).reshape(0,self.angles)
        row_poses = np.array([]).reshape(0,self.col,self.angles)

        assert len(self.img_paths) == self.row*self.col*self.angles

        for impath in self.img_paths:
            print("row_idx",row_idx," col_idx",col_idx," angle_idx",angle_idx)
            grayim = self.read_image(impath)
            keypoints, descriptors = self.superpoint_generator(grayim)

            image_pose = ImagePose(self.calibration)
            image_pose.kp = keypoints
            image_pose.desc = descriptors
            angle_poses = np.hstack((angle_poses, image_pose)) 
            angle_idx += 1
            if(angle_idx == self.angles):
                angle_idx = 0
                col_poses = np.vstack((col_poses, np.expand_dims(angle_poses,axis=0)))
                col_idx += 1
                if(col_idx == self.col):
                    col_idx = 0
                    row_idx += 1
                    row_poses = np.vstack((row_poses, np.expand_dims(col_poses,axis=0)))
                    col_poses = np.array([]).reshape(0,self.angles)
                angle_poses = np.array([])



        self.img_pose = row_poses


    def save_cvs_file(self, kp_data, desc_data, pose_desc_data, i ,j):
        """This is used to save the features information of each frame into a .csv file,
        kp: matched key points
        desc: matched descriptors
        pose desc: All descriptors of a certain image 

        [N*2 k1, N*2 k2, N*256 d1, N*256 d2]"""

        np.savetxt('datasets/4.5/mapping/kp/%d_'% i+'%d_keypoints.csv' % j,kp_data, fmt='%10.2f')
        np.savetxt('datasets/4.5/mapping/desc/%d_'% i+'%d_descriptors.csv' % j,desc_data,fmt='%10.5f',delimiter=',')
        np.savetxt('datasets/4.5/mapping/kp/descriptors_%d.csv' % i,pose_desc_data,fmt='%10.5f',delimiter=',')

    def read_kp_cvs_file(self, file_path):
        return np.genfromtxt(file_path)

    def read_desc_cvs_file(self, file_path):
        return np.genfromtxt(file_path,delimiter=',')

    def save_atrium_map(self, landmark_points, descriptors, poses):
        return 

    def read_atrium_map(self, file_path):
        return np.genfromtxt(file_path,delimiter=',')

    def feature_matching(self,save = False):
        """
        Feature will be matched based on:
            - L2 distances of descriptors
        """
        for m in range(self.angles):
            # Iterate through all images.
            for i in range(0,self.directions*self.row*self.col-1):
                for j in range(i+1,self.directions*self.row*self.col):  
                    ia,i_mod = divmod(i,self.row*self.col) 
                    ir, ic = divmod(i_mod, self.col)
                    ia = m+ia
                    if ia == 8: ia = 0
                    if ia == 9: ia = 1
                    ja,j_mod = divmod(j,self.row*self.col) 
                    jr, jc = divmod(j_mod, self.col)
                    ja = m+ja
                    if ja == 8: ja = 0
                    if ja == 9: ja = 1

                    matches = self.point_tracker.nn_match_two_way(self.img_pose[ir][ic][ia].desc, self.img_pose[jr][jc][ja].desc, self.nn_thresh)
                    matches = matches[:,matches[2,:].argsort()[::-1]]
                    matches=matches[:,matches[2]>0.4]

                    # size is N*2
                    src_points = self.img_pose[ir][ic][ia].kp[matches[0].astype(int)]
                    dst_points = self.img_pose[jr][jc][ja].kp[matches[1].astype(int)]

                    good_match_count = 0
                    for k in range(matches.shape[1]):
                        if self.img_pose[ir][ic][ia].kp_matches.get(matches[0,k]) is None:
                            self.img_pose[ir][ic][ia].kp_matches[matches[0,k]] = {}
                        self.img_pose[ir][ic][ia].kp_matches[matches[0,k]][jr,jc,ja] = matches[1,k]
                        if self.img_pose[jr][jc][ja].kp_matches.get(matches[1,k]) is None:
                            self.img_pose[jr][jc][ja].kp_matches[matches[1,k]] = {}
                        self.img_pose[jr][jc][ja].kp_matches[matches[1,k]][ir,ic,ia] = matches[0,k]
                        good_match_count+=1

                    print("Feature matching ",ir, ic, ia," ",jr, jc, ja, " ==> ", good_match_count, "/" ,len(self.img_pose[ir][ic][ia].desc.T))
        
        if save:
            # size is N*256
            src_descriptors = np.transpose(self.img_pose[ir][ic][ia].desc)[matches[0].astype(int)]
            dst_descriptors = np.transpose(self.img_pose[jr][jc][ja].desc)[matches[1].astype(int)]

            # Output data [N*2 k1, N*2 k2, N*2 matches, N*256 d1, N*256 d2]
            # kp_data =  np.hstack((src_points,dst_points,np.transpose(matches[:2])))
            kp_data =  np.hstack((src_points,dst_points,matches.T))
            desc_data = np.hstack((src_descriptors,dst_descriptors))
            pose_desc_data =  np.transpose(self.img_pose[ir][ic][ia].desc)
            # Save to cvs files
            self.save_cvs_file(kp_data,desc_data,pose_desc_data,i,j)

    def initial_estimation(self):
        # for i in range(self.row*self.col*self.angles-1):
            # for j in range(i+1,self.row*self.col*self.angles):  
            # ir,i_mod = divmod(i,self.angles*self.col) 
            # ic, ia = divmod(i_mod, self.angles)
            # j = i+1
            # jr,j_mod = divmod(j,self.angles*self.col) 
            # jc, ja = divmod(j_mod, self.angles)
            # src = np.array([], dtype=np.float).reshape(0,2)
            # dst = np.array([], dtype=np.float).reshape(0,2)
            # kp_src_idx = []
            # kp_dst_idx = []
        for n in range(self.angles):
            # Iterate through all images.
            for i in range(0,self.directions*self.row*self.col-1):
                for j in range(i+1,self.directions*self.row*self.col):  
                    ia,i_mod = divmod(i,self.row*self.col) 
                    ir, ic = divmod(i_mod, self.col)
                    ia = n+ia
                    if ia == 8: ia = 0
                    if ia == 9: ia = 1
                    ja,j_mod = divmod(j,self.row*self.col) 
                    jr, jc = divmod(j_mod, self.col)
                    ja = n+ja
                    if ja == 8: ja = 0
                    if ja == 9: ja = 1
     

                    src = np.array([], dtype=np.float).reshape(0,2)
                    dst = np.array([], dtype=np.float).reshape(0,2)
                    kp_src_idx = []
                    kp_dst_idx = []

                    for k in self.img_pose[ir][ic][ia].kp_matches:
                        if self.img_pose[ir][ic][ia].kp_match_exist(k,(jr,jc,ja)):
                            k = int(k)
                            match_idx = self.img_pose[ir][ic][ia].kp_match_idx(k,(jr,jc,ja))
                            match_idx = int(match_idx)

                            src = np.vstack((src,self.img_pose[ir][ic][ia].kp[k]))
                            dst = np.vstack((dst,self.img_pose[jr][jc][ja].kp[match_idx]))
                            kp_dst_idx.append(match_idx)
                            kp_src_idx.append(k)

                    if src.shape[0] < 6:
                        print("Not enough points to generate essential matrix for image_",ir, ic, ia," and image_",jr, jc, ja)
                        continue


                    src = np.expand_dims(src, axis=1)
                    dst = np.expand_dims(dst, axis=1) 
 
                    E, mask = cv2.findEssentialMat(dst,src,cameraMatrix = self.calibration,method =cv2.LMEDS,prob=0.999)
                    # E, mask = cv2.findEssentialMat(dst,src,cameraMatrix = self.calibration,method =cv2.RANSAC,prob=0.999, threshold=1)
                    new_src = np.array([], dtype=np.float).reshape(0,2)
                    new_dst = np.array([], dtype=np.float).reshape(0,2)
                    new_kp_src_idx = []
                    for idx,check in enumerate(mask):
                        if check > 0:
                            new_src = np.vstack((new_src,src[idx]))
                            new_dst = np.vstack((new_dst,dst[idx]))
                            new_kp_src_idx.append(kp_src_idx[idx])
                    dst = np.expand_dims(new_dst, axis=1) 
                    src = np.expand_dims(new_src, axis=1) 
                    kp_src_idx = new_kp_src_idx   
                    _, local_R, local_t, mask=cv2.recoverPose(E,dst,src,cameraMatrix = self.calibration)

                    T = Pose3(Rot3(local_R),Point3(local_t[0],local_t[1],local_t[2]))

                    self.img_pose[jr][jc][ja].T = transform_from(T,self.img_pose[ir][ic][ia].T)
                    cur_R = self.img_pose[jr][jc][ja].T.rotation().matrix()
                    cur_t = self.img_pose[jr][jc][ja].T.translation().vector()
                    cur_t = np.expand_dims(cur_t, axis=1)
                    self.img_pose[jr][jc][ja].P = np.dot(self.calibration,np.hstack((cur_R.T,-np.dot(cur_R.T,cur_t))))

                    points4D = cv2.triangulatePoints(projMatr1=self.img_pose[ir][ic][ia].P,projMatr2=self.img_pose[jr][jc][ja].P,projPoints1=src, projPoints2=dst)
                    points4D = points4D / np.tile(points4D[-1, :], (4, 1))
                    pt3d = points4D[:3, :].T

                    # Find good triangulated points
                    for m,k in enumerate(kp_src_idx):
                        if(mask[m]):
                            match_idx = self.img_pose[ir][ic][ia].kp_match_idx(k, (jr,jc,ja))
                            k = int(k)
                            match_idx = int(match_idx)
                            if (self.img_pose[ir][ic][ia].kp_3d_exist(k)):
                                self.img_pose[jr][jc][ja].kp_landmark[match_idx] = self.img_pose[ir][ic][ia].kp_3d(k)
                                self.landmark[self.img_pose[ir][ic][ia].kp_3d(k)].point += pt3d[m]
                                # if (jr-ir)**2 + (jc-ic)**2>=4:
                                self.landmark[self.img_pose[jr][jc][ja].kp_3d(match_idx)].seen += 1
                                self.landmark[self.img_pose[ir][ic][ia].kp_3d(k)].desc += self.img_pose[jr][jc][ja].desc.T[match_idx]

                            else:
                                new_landmark = LandMark(pt3d[m],2)
                                new_landmark.desc = self.img_pose[ir][ic][ia].desc.T[k]+self.img_pose[jr][jc][ja].desc.T[match_idx]
                                self.landmark.append(new_landmark)

                                self.img_pose[ir][ic][ia].kp_landmark[k] = len(self.landmark) - 1
                                self.img_pose[jr][jc][ja].kp_landmark[match_idx] = len(self.landmark) - 1

                for m in range(len(self.landmark)):
                    if(self.landmark[m].seen>=3):
                        self.landmark[m].point/=(self.landmark[m].seen - 1)
                        self.landmark[m].desc /= self.landmark[m].seen
                        self.landmark[m].desc = self.landmark[m].desc/np.linalg.norm(self.landmark[m].desc)

    def back_projection(self, key_point = Point2(), pose = Pose3(), depth = 10):
        # Normalize input key_point
        pn = self.cal.calibrate(key_point)

        # Transfer normalized key_point into homogeneous coordinate and scale with depth
        ph = Point3(depth*pn.x(),depth*pn.y(),depth)

        # Transfer the point into the world coordinate
        return pose.transform_from(ph)

    def create_pose_initial_estimation(self):
        theta = np.radians(30)
        wRc = np.array([[-math.sin(theta), 0 , math.cos(theta)],
                        [-math.cos(theta), 0 , -math.sin(theta)],
                        [0, -1 , 0]])
        delta = np.radians(45)
        R = Rot3(np.array([[math.cos(delta), 0 , math.cos(delta)],
                        [0, 1 , 0],
                        [-math.sin(delta), 0 , math.sin(delta)]]))
        delta_x = delta_y = 1.25

        row_idx = 0
        col_idx = 0
        angle_idx = 0
        angle_poses = np.array([])
        col_poses = np.array([]).reshape(0,self.angles)
        row_poses = np.array([]).reshape(0,self.col,self.angles)

        for i in range(self.row*self.col*self.angles):
            image_pose = Pose3(Rot3(np.dot(wRc,
                        # cRc'
                        np.array([[math.cos(delta*angle_idx), 0 , -math.sin(delta*angle_idx)],
                        [0, 1 , 0],
                        [math.sin(delta*angle_idx), 0 , math.cos(delta*angle_idx)]])
                        )),
                        Point3(-math.sin(theta)*delta_x*col_idx+math.cos(theta)*delta_y*row_idx,-math.cos(theta)*delta_x*col_idx-math.sin(theta)*delta_y*row_idx,1.2))
                        # Point3(-math.sin(theta)*delta_x*col_idx, -math.cos(theta)*delta_x*col_idx, 1.2))
                        # Point3(-math.sin(theta)*1.58*col_idx, -math.cos(theta)*1.58*col_idx, 1.2))
            angle_poses = np.hstack((angle_poses, image_pose))            
            angle_idx += 1
            if(angle_idx == self.angles):
                angle_idx = 0
                col_poses = np.vstack((col_poses, np.expand_dims(angle_poses,axis=0)))
                col_idx+=1
                if(col_idx == self.col):
                    col_idx = 0
                    row_idx += 1
                    row_poses = np.vstack((row_poses, np.expand_dims(col_poses,axis=0)))
                    col_poses = np.array([]).reshape(0,self.angles)
                angle_poses = np.array([])
            

        self.pose_initial = row_poses


    def bundle_adjustment(self):
        MIN_LANDMARK_SEEN = 3

        depth = 15

        # Initialize factor Graph
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()

        # Add factors for all measurements
        measurementNoiseSigma = 1.0
        measurementNoise = gtsam.noiseModel_Isotropic.Sigma(
            2, measurementNoiseSigma)

        landmark_id_list = {}
        img_pose_id_list = {}
        for i in range(self.row*self.col*self.angles):
            ir,i_mod = divmod(i,self.angles*self.col) 
            ic, ia = divmod(i_mod, self.angles)
            img_pose = self.img_pose[ir][ic][ia]
            for k in range(len(img_pose.kp)):
                if img_pose.kp_3d_exist(k):
                    landmark_id = img_pose.kp_3d(k)
                    if(self.landmark[landmark_id].seen >= MIN_LANDMARK_SEEN):
                        key_point = Point2(img_pose.kp[k][0],img_pose.kp[k][1])
                        if landmark_id_list.get(landmark_id) == None:
                            pose = self.pose_initial[ir][ic][ia]
                            landmark_point = self.back_projection(key_point,pose,depth)
                            landmark_id_list[landmark_id] = landmark_point
                            # print("careful",ir,ic,ia,pose,landmark_point)
                        if img_pose_id_list.get(i) == None:
                            img_pose_id_list[i] = True           
                        graph.add(gtsam.GenericProjectionFactorCal3_S2(
                        key_point, measurementNoise,
                        X(i), P(landmark_id), self.cal))

        s = np.radians(60)
        poseNoiseSigmas = np.array([s, s, s, 10, 10, 10])
        # poseNoiseSigmas = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        posePriorNoise = gtsam.noiseModel_Diagonal.Sigmas(poseNoiseSigmas)
        print(img_pose_id_list)
        for i,idx in enumerate(img_pose_id_list): 
            ir,i_mod = divmod(idx,self.angles*self.col) 
            ic, ia = divmod(i_mod, self.angles)
            # Initialize estimate for poses
            pose_i = self.pose_initial[ir][ic][ia]
            initialEstimate.insert(X(idx), pose_i)
            # Create priors for poses
            # counter = 0
            # if(counter == 0 and ia == 4):
            #     temp_a = ia
            #     temp_r = ir
            #     temp_c = ic
            #     graph.add(gtsam.PriorFactorPose3(X(idx), pose_i, posePriorNoise))
            #     counter+=1 
            # if(counter == 1 and temp_a == ia and (temp_r-ir)**2 + (temp_c-ic)**2>=4):
            #     graph.add(gtsam.PriorFactorPose3(X(idx), pose_i, posePriorNoise)) 
            #     counter +=1
            # if i <2:
            #     graph.add(gtsam.PriorFactorPose3(X(idx), pose_i, posePriorNoise)) 
            graph.add(gtsam.PriorFactorPose3(X(idx), pose_i, posePriorNoise)) 

        # Initialize estimate for landmarks
        print(landmark_id_list)
        for idx in landmark_id_list:
            point_j = landmark_id_list.get(idx)
            initialEstimate.insert(P(idx), point_j)
            self.landmark_descriptor.append(self.landmark[idx].desc)

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        sfm_result = optimizer.optimize()
        # Marginalization
        marginals = gtsam.Marginals(graph, sfm_result)
        for idx in img_pose_id_list: 
            marginals.marginalCovariance(X(idx))
        for idx in landmark_id_list: 
            marginals.marginalCovariance(P(idx))

        return sfm_result,img_pose_id_list,landmark_id_list

    def plot_sfm_result(self,result):

        # Declare an id for the figure
        fignum = 0

        fig = plt.figure(fignum)
        axes = fig.gca(projection='3d')
        plt.cla()

        # Plot points
        # Can't use data because current frame might not see all points
        # marginals = Marginals(isam.getFactorsUnsafe(), isam.calculateEstimate())
        # gtsam.plot_3d_points(result, [], marginals)
        gtsam_plot.plot_3d_points(fignum, result, 'rx')
        # gtsam_plot.plot_pose3(fignum, result, 'rx')

        # Plot cameras
        i = 0
        while result.exists(X(i)):
            pose_i = result.atPose3(X(i))
            gtsam_plot.plot_pose3(fignum, pose_i, 2)
            i += 1

        # draw
        axes.set_xlim3d(-50, 50)
        axes.set_ylim3d(-50, 50)
        axes.set_zlim3d(-50, 50)
        plt.legend()
        plt.show()
    
    def create_atrium_map(self,sfm_result,img_pose_id_list,landmark_id_list):
        """
        Create a atrium map .
        """
        atrium_trajectory = []
        initial_pose = sfm_result.atPose3(X(img_pose_id_list[0]))
        atrium_trajectory.append(initial_pose)

        atrium_points = []
        for idx in landmark_id_list:
            point_j = sfm_result.atPoint3(P(idx))
            atrium_points.append(point_j)

        atrium_descriptors = np.array(self.landmark_descriptor)
        
        atrium_map = Map(np.array(atrium_points), np.array(atrium_descriptors), atrium_trajectory)

        return atrium_map

    def sim3_transform_map(self, result, img_pose_id_list, landmark_id_list):
        # Similarity transform
        s_poses = []
        s_points = []
        _sim3 = sim3.Similarity3()

        theta = np.radians(30)
        wRc = gtsam.Rot3(np.array([[-math.sin(theta), 0, math.cos(theta)],
                                [-math.cos(theta), 0, -math.sin(theta)],
                                [0, -1, 0]]))

        pose_pairs = []
        for i,idx in enumerate(img_pose_id_list):
            pose_i = result.atPose3(symbol(ord('x'), idx))
            s_poses.append(pose_i)
            if i < 2:
                d_pose = gtsam.Pose3(
                    wRc, gtsam.Point3(-math.sin(theta)*1.58*idx, -math.cos(theta)*1.58*idx, 1.2))
                pose_pairs.append([s_poses[i],d_pose])
                i+=1


        for idx in landmark_id_list:
            point_j = result.atPoint3(symbol(ord('p'), idx))
            s_points.append(point_j)


        _sim3.align_pose(pose_pairs)
        print('R', _sim3._R)
        print('t', _sim3._t)
        print('s', _sim3._s)

        s_map = (s_poses, s_points)
        actual_poses, actual_points = _sim3.map_transform(s_map)
        d_map = _sim3.map_transform(s_map)

        return _sim3, d_map

    def print_sim3_result(self, d_map):
        for i,point in enumerate(d_map[1]):
            print('Point_',i,':',point)
        for i,pose in enumerate(d_map[0]):
            print('Pose_',i,':',pose)


    def plot_sim3_result(self, fignum, result):

        fig = plt.figure(fignum)
        axes = fig.gca(projection='3d')
        plt.cla()

        # Plot points
        # Can't use data because self.img_pose[jr][jc][ja]rent frame might not see all points
        # marginals = Marginals(isam.getFactorsUnsafe(), isam.calculateEstimate())
        # gtsam.plot_3d_points(result, [], marginals)
        for point in result[1]:
            gtsam_plot.plot_point3(fignum, point, 'rx')

        # Plot cameras
        for pose in result[0]:
            gtsam_plot.plot_pose3(fignum, pose, 1)
        # draw
        axes.set_xlim3d(-20, 20)
        axes.set_ylim3d(-20, 20)
        axes.set_zlim3d(-20, 20)
        plt.legend()
        plt.show()

    def angle_between(self,v1, v2):
        """
        Input:
            Two n dimensional normalized vectors
        Output:
            The angle between the two vectors.
        """
        # Refer to https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
        angle = np.math.atan2(np.linalg.det([v1,v2]),np.dot(v1,v2))
        return angle

    def plot_single_img_descriptors(self):
        
        for i in range(5):
            fig = plt.figure(1)
            x = np.linspace(-2,2)
            y = x
            plt.plot(x, y)
            descriptors = self.img_pose[i].desc.T
            desc0 = descriptors[0]
            for desc in descriptors:
                theta = self.angle_between(desc,desc0)
                newline([0,0],[math.cos(theta),math.sin(theta)])
            plt.legend()
            plt.show()

        

    def plot_diff_landmark_corresponding_desc_in_all_img(self):
        return

if __name__ == "__main__":
#     np.set_printoptions(suppress=True,
#    formatter={'float_kind':'{:0.2f}'.format})
    fe = MappingFrontEnd()

    # undistort_image('datasets/5x5x8/', '*.jpg')
    fe.get_image_paths()
    fe.extract_all_image_features()
    fe.feature_matching()
    fe.initial_estimation()
    fe.create_pose_initial_estimation()

    tic_ba = time.time()    
    sfm_result,nrCamera, nrPoint = fe.bundle_adjustment()
    toc_ba = time.time()
    print(sfm_result)
    print('BA spents ',toc_ba-tic_ba,'s')
    fe.plot_sfm_result(sfm_result)

    # Similarity Transform
    # _, result =fe.sim3_transform_map(sfm_result, nrCamera, nrPoint)
    # fe.print_sim3_result(result)
    # fe.plot_sim3_result(0, result)