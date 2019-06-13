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
import time
from sfm import sim3
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import statistics as st
from scipy import stats

sys.path.append('../')
# https://gist.github.com/abidrahmank/6450018

def X(i):
    """Create key for pose i."""
    return symbol(ord('x'), i)


def P(j):
    """Create key for landmark j."""
    return symbol(ord('p'), j)


def transform_from(T, pose):
    """ Calculate the Euclidean transform of a Pose3
    Parameters:(
        pose - Pose3 object
    Returns:
        Pose3 object
    """

    R = T.rotation().compose(pose.rotation())
    translation = T.rotation().rotate(pose.translation()).vector() + T.translation().vector()
    return Pose3(R, Point3(translation))

def undistort_image(basedir,img_extension):
    search = os.path.join(basedir, img_extension)
    img_paths = glob.glob(search)
    img_paths.sort()
    print("Number of Images: ",len(img_paths))
    maxlen = len(img_paths)
    if maxlen == 0:
        raise IOError('No images were found (maybe wrong \'image extension\' parameter?)')

    calibration = gtsam.Cal3_S2(fx=333.4, fy=314.7,s=0,u0=303.6, v0=247.6).matrix()
    distortion = np.array([-0.282548, 0.054412, -0.001882, 0.004796, 0.000000])
    for i,img_path in enumerate(img_paths):
        img = cv2.imread(img_path, 0)
        dst = cv2.undistort(img,calibration ,distortion , None, calibration)
        # cv2.imshow(dst)
        output_path = '%d_frame_undist.jpg'%i
        output_path = os.path.join(basedir, output_path)
        cv2.imwrite(output_path,dst)

class ImagePose(object):
    def __init__(self,calibration):
        self.desc = np.array([])
        self.kp = np.array([])
        self.R = Rot3()
        self.t = Point3()
        self.T = Pose3(self.R,self.t)
        self.P = np.dot(calibration,np.hstack((np.identity(3),np.zeros((3,1)))))
        self.kp_matches = {}
        self.kp_landmark = {}
    
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
    def __init__(self,point=np.array([0,0,0]),seen = 0):
        self.point = point
        self.seen = seen

class MappingFrontEnd(object):
    def __init__(self,image_directory_path = 'datasets/4.5/mapping/source_images/', image_extension = '*.jpg', image_size = (640,480), down_sample = 1, nn_thresh = 0.7):
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

        # fov_in_degrees = 128, image_width = 640, image_height = 480)
        # fx = w/2*math.tan(fov*3.14/360)
        # How resizing an image affect the intrinsic camera matrix:https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
        self.image_transform = np.array([[0.25,0,0.125-0.5],[0,0.25,0.125-0.5],[0,0,1]])
        # self.calibration = gtsam.Cal3_S2(fx=156.3, fy=236,s=0,u0=320, v0=240).matrix()
        self.calibration = gtsam.Cal3_S2(fx=333.4, fy=314.7,s=0,u0=303.6, v0=247.6).matrix()
        # self.calibration = np.dot(self.image_transform,self.calibration)
        self.distortion = np.array([-0.282548, 0.054412, -0.001882, 0.004796, 0.000000])
        self.rectification = np.identity(3)
        self.projection = np.array([[226.994629 ,0.000000, 311.982613, 0.000000],[0.000000, 245.874146, 250.410089, 0.000000],[0.000000, 0.000000, 1.000000, 0.000000]])
        # self.projection = np.dot(self.image_transform,self.projection)
        # self.cal = gtsam.Cal3_S2(fx=156.3, fy=236,s=0,u0=320, v0=240)
        # self.cal = gtsam.Cal3_S2(fx=19.54, fy=29.5,s=0,u0=39.625, v0=29.625)
        # self.cal = gtsam.Cal3_S2(fx=83.35, fy=78.675,s=0,u0=75.65, v0=61.65)
        self.cal= gtsam.Cal3_S2(fx=333.4, fy=314.7,s=0,u0=303.6, v0=247.6)

        
        self.keypoint_list = []
        self.descriptor_list = []
        self.correspondence_table = []
        
        self.img_pose=[]
        self.landmark=[]




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
        """"""
        for impath in self.img_paths:
            grayim = self.read_image(impath)
            keypoints, descriptors = self.superpoint_generator(grayim)
            image_pose = ImagePose(self.calibration)
            image_pose.kp = keypoints
            image_pose.desc = descriptors
            self.img_pose.append(image_pose)


    def save_cvs_file(self, kp_data, desc_data, pose_desc_data, i ,j):
        """This is used to save the features information of each frame into a .csv file,
        [N*2 k1, N*2 k2, N*256 d1, N*256 d2]"""

        np.savetxt('datasets/4.5/mapping/kp/%d_'% i+'%d_keypoints.csv' % j,kp_data, fmt='%d')
        np.savetxt('datasets/4.5/mapping/desc/%d_'% i+'%d_descriptors.csv' % j,desc_data,fmt='%10.5f',delimiter=',')
        np.savetxt('datasets/4.5/mapping/kp/descriptors_%d.csv' % i,pose_desc_data,fmt='%10.5f',delimiter=',')

    def read_kp_cvs_file(self, file_path):
        return np.genfromtxt(file_path)

    def read_desc_cvs_file(self, file_path):
        return np.genfromtxt(file_path,delimiter=',')

    def feature_matching(self,save = False):
        """
        
        Feature will be matched based on:
            - L2 distances of descriptors
            - findFundamentalMat
        """


        # Iterate through all images.
        for i in range(0,len(self.img_pose)-1):
            for j in range(i+1,len(self.img_pose)):
                matches = self.point_tracker.nn_match_two_way(self.img_pose[i].desc, self.img_pose[j].desc, self.nn_thresh)

                # size is N*2
                src_points = self.img_pose[i].kp[matches[0].astype(int)]
                dst_points = self.img_pose[j].kp[matches[1].astype(int)]

                F_, inliers= cv2.findFundamentalMat(src_points,dst_points,cv2.FM_RANSAC,3,0.99)

                good_match_count = 0
                for k,good_match in enumerate(inliers):

                    if self.img_pose[i].kp_matches.get(matches[0,k]) is None:
                        self.img_pose[i].kp_matches[matches[0,k]] = {}
                    self.img_pose[i].kp_matches[matches[0,k]][j] = matches[1,k]
                    if self.img_pose[j].kp_matches.get(matches[1,k]) is None:
                        self.img_pose[j].kp_matches[matches[1,k]] = {}
                    self.img_pose[j].kp_matches[matches[1,k]][i] = matches[0,k]
                    good_match_count+=1

                print("Feature matching ",i," ",j, " ==> ", good_match_count, "/" ,len(self.img_pose[i].desc.T))

                # size is N*256
                src_descriptors = np.transpose(self.img_pose[i].desc)[matches[0].astype(int)]
                dst_descriptors = np.transpose(self.img_pose[j].desc)[matches[1].astype(int)]

                # Output data [N*2 k1, N*2 k2, N*2 matches, N*256 d1, N*256 d2]
                kp_data =  np.hstack((src_points,dst_points,np.transpose(matches[:2])))
                desc_data = np.hstack((src_descriptors,dst_descriptors))
                pose_desc_data =  np.transpose(self.img_pose[i].desc)
        
        if save:
            # Save to cvs files
            self.save_cvs_file(kp_data,desc_data,pose_desc_data,i,j)

    def initial_estimation(self):
        for i in range(len(self.img_pose)-1):
        # for i in range(2):
            src = np.array([], dtype=np.float).reshape(0,2)
            dst = np.array([], dtype=np.float).reshape(0,2)
            kp_src_idx = []
            kp_dst_idx = []


            for k in self.img_pose[i].kp_matches:
                if self.img_pose[i].kp_match_exist(k,i+1):
                    k = int(k)
                    
                    match_idx=self.img_pose[i].kp_match_idx(k,i+1)
                    match_idx = int(match_idx)
                    
                    src = np.vstack((src,self.img_pose[i].kp[k]))
                    dst = np.vstack((dst,self.img_pose[i+1].kp[match_idx]))
                    kp_dst_idx.append(match_idx)
                    kp_src_idx.append(k)


            if src.shape[0] < 6:
                print("Not enough points to generate essential matrix for image_",i, " and image_", i+1)
                continue
            src_norm = cv2.undistortPoints(np.expand_dims(src, axis=1), cameraMatrix=self.calibration, distCoeffs=self.distortion, P = self.projection)
            dst_norm = cv2.undistortPoints(np.expand_dims(dst, axis=1), cameraMatrix=self.calibration, distCoeffs=self.distortion, P = self.projection)              
            # src_norm = cv2.undistortPoints(np.expand_dims(src, axis=1), cameraMatrix=self.calibration, distCoeffs=self.distortion)
            # dst_norm = cv2.undistortPoints(np.expand_dims(dst, axis=1), cameraMatrix=self.calibration, distCoeffs=self.distortion)   
            E, mask = cv2.findEssentialMat(dst_norm,src_norm,cameraMatrix = self.calibration,method =cv2.RANSAC,prob=0.999,threshold=0.1)

     

            # _, local_R, local_t, mask=cv2.recoverPose(E,dst_norm,src_norm)

            theta = np.radians(30)
            local_R = np.array([[-math.sin(theta), 0 , math.cos(theta)],
                        [-math.cos(theta), 0 , -math.sin(theta)],
                        [0, 1 , 0]])
            local_t = np.array([[-math.sin(theta)*1.58], [-math.cos(theta)*1.58], [0]])

            # new_src = np.array([], dtype=np.float).reshape(0,2)
            # new_dst = np.array([], dtype=np.float).reshape(0,2)
            # new_kp_src_idx = []
            # for idx,check in enumerate(mask):
            #     if check > 0:
            #         new_src = np.vstack((new_src,src_norm[idx]))
            #         new_dst = np.vstack((new_dst,dst_norm[idx]))
            #         new_kp_src_idx.append(kp_src_idx[idx])
            # dst_norm = new_dst 
            # src_norm = new_src 
            # kp_src_idx = new_kp_src_idx

            T = Pose3(Rot3(local_R),Point3(local_t[0],local_t[1],local_t[2]))

            self.img_pose[i+1].T = transform_from(T,self.img_pose[i].T)

            # Keep in mind that OpenCV defines the rotation and translation in the direction the points move, not the direction the camera moves. 
            self.img_pose[i+1].P = np.hstack((local_R.T,-np.dot(local_R.T,local_t)))

            # self.img_pose[i+1].P = np.hstack((self.img_pose[i+1].T.rotation().matrix(),self.img_pose[i+1].T.translation().vector().T))
    
            self.img_pose[i+1].P = np.dot(self.calibration,self.img_pose[i+1].P)
            points4D = cv2.triangulatePoints(projMatr1=self.img_pose[i].P,projMatr2=self.img_pose[i+1].P,projPoints1=src_norm, projPoints2=dst_norm)
            points4D = points4D / np.tile(points4D[-1, :], (4, 1))
            pt3d = points4D[:3, :].T

            #  Scale the new 3d points to be similar to the existing 3d points (landmark)
            #  Use ratio of distance between pairing 3d points
            if i>0:
                scale = 0
                count = 0

                new_pts = []
                existing_pts = []

                for j,k in enumerate(kp_src_idx):
                    if(mask[j] & self.img_pose[i].kp_match_exist(k,i+1) & self.img_pose[i].kp_3d_exist(k)):
                    
                        idx = self.img_pose[i].kp_3d(k)
                        avg_landmark = self.landmark[idx].point / (self.landmark[idx].seen -1)
                        new_pts.append(pt3d[j])
                        existing_pts.append(avg_landmark)
                
                _x = []
                _y = []
                for j in range(len(new_pts)-1):
                    for k in range(j+1,len(new_pts)):
                        s = np.linalg.norm(existing_pts[j] - existing_pts[k]) / np.linalg.norm(new_pts[j] - new_pts[k])
                        scale += s
                        count += 1
                #         _x.append(np.linalg.norm(new_pts[j] - new_pts[k]))
                #         _y.append(np.linalg.norm(existing_pts[j] - existing_pts[k]))
                # x = np.append(np.array(_x), [0], axis=0)
                # y = np.append(np.array(_y), [0], axis=0)
                
                # print(st.median(_y))
                # print(x)
                # print(y)
                # A = np.vstack([x, np.ones(len(x))]).T
                # m, c = np.linalg.lstsq(A, y, rcond=-1)[0]
                # s,intercept,_,_,_ = stats.linregress(x,y)
                # scale = s
                # print(st.median(x),st.median(y),scale)
                # plt.plot(x, y, 'o', label='original data')
                # plt.plot(x, intercept + s*x, 'r', label='fitted line')
                # plt.legend()
                # plt.show()
                # assert count > 0, "Count=0, Not enough matches"
                if count == 0:
                    print("count =0")
                    continue
                # scale /= count
                scale =1
                

                print("image ",i+1," ==>",i, " scale = ", scale, "count=" ,count)

                #  apply scale and re-calculate T and P matrix
                local_t *= scale

                T = Pose3(Rot3(local_R),Point3(local_t[0],local_t[1],local_t[2]))
                self.img_pose[i+1].T = transform_from(T,self.img_pose[i].T)

                # Keep in mind that OpenCV defines the rotation and translation in the direction the points move, not the direction the camera moves. 
                self.img_pose[i+1].P = np.hstack((local_R.T,-np.dot(local_R.T,local_t)))
                self.img_pose[i+1].P = np.dot(self.calibration,self.img_pose[i+1].P)
                points4D = cv2.triangulatePoints(projMatr1=self.img_pose[i].P,projMatr2=self.img_pose[i+1].P,projPoints1=src_norm, projPoints2=dst_norm)
                points4D = points4D / np.tile(points4D[-1, :], (4, 1))
                pt3d = points4D[:3, :].T

            # Find good triangulated points
            for j,k in enumerate(kp_src_idx):
                if(mask[j]):
                    match_idx = self.img_pose[i].kp_match_idx(k, i+1)

                    if (self.img_pose[i].kp_3d_exist(k)):
                        self.img_pose[i+1].kp_landmark[match_idx] = self.img_pose[i].kp_3d(k)
                        self.landmark[self.img_pose[i].kp_3d(k)].point += pt3d[j]
                        self.landmark[self.img_pose[i+1].kp_3d(match_idx)].seen += 1

                    else:
                        new_landmark = LandMark(pt3d[j],2)
                        self.landmark.append(new_landmark)

                        self.img_pose[i].kp_landmark[k] = len(self.landmark) - 1
                        self.img_pose[i+1].kp_landmark[match_idx] = len(self.landmark) - 1

        for j in range(len(self.landmark)):
            if(self.landmark[i].seen>=3):
                self.landmark[i].point/=(self.landmark[i].seen - 1)

    def bundle_adjustment(self):
        MIN_LANDMARK_SEEN = 3
 
        # Initialize factor Graph
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()

        # Add factors for all measurements
        measurementNoiseSigma = 1.0
        measurementNoise = gtsam.noiseModel_Isotropic.Sigma(
            2, measurementNoiseSigma)

        landmark_id_list = {}
        img_pose_id_list = {}
        for i,img_pose in enumerate(self.img_pose):
            for k in range(len(img_pose.kp)):
                if img_pose.kp_3d_exist(k):
                    landmark_id = img_pose.kp_3d(k)
                    if(self.landmark[landmark_id].seen >= MIN_LANDMARK_SEEN):
                        landmark_id_list[landmark_id] = True  
                        img_pose_id_list[i] = True           
                        graph.add(gtsam.GenericProjectionFactorCal3_S2(
                        Point2(img_pose.kp[k][0],img_pose.kp[k][1]), measurementNoise,
                        X(i), P(landmark_id), self.cal))

            
        # Initialize estimate for poses
        # Create priors 
        s = np.radians(60)
        poseNoiseSigmas = np.array([s, s, s, 10, 10, 10])
        # poseNoiseSigmas = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        posePriorNoise = gtsam.noiseModel_Diagonal.Sigmas(poseNoiseSigmas)
        
        theta = np.radians(30)
        wRc = gtsam.Rot3(np.array([[-math.sin(theta), 0 , math.cos(theta)],
                        [-math.cos(theta), 0 , -math.sin(theta)],
                        [0, 1 , 0]]))

        for i,idx in enumerate(img_pose_id_list):   
            initialEstimate.insert(X(idx), self.img_pose[idx].T)
            if i < 2:
                graph.add(gtsam.PriorFactorPose3(X(idx), gtsam.Pose3(
                wRc, gtsam.Point3(-math.sin(theta)*1.58*idx, -math.cos(theta)*1.58*idx, 1.2)), posePriorNoise)) 
                # i+=1

        # Initialize estimate for landmarks
        for idx in landmark_id_list:
            p = self.landmark[idx].point

            initialEstimate.insert(P(idx), Point3(p[0],p[1],p[2]))

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        # sfm_result = optimizer.optimize()
        for i in range(1000):
            optimizer.iterate()
        sfm_result = optimizer.values()
        # print(sfm_result)
        # Marginalization
        # marginals = gtsam.Marginals(graph, sfm_result)
        # for idx in img_pose_id_list: 
        #     marginals.marginalCovariance(X(idx))
        # for idx in landmark_id_list: 
        #     marginals.marginalCovariance(P(idx))


        nrCamera = len(img_pose_id_list)
        nrPoint = len(landmark_id_list)

        return sfm_result,img_pose_id_list,landmark_id_list

    def sim3_transform_map(self, result, img_pose_id_list, landmark_id_list):
        # Similarity transform
        s_poses = []
        s_points = []
        _sim3 = sim3.Similarity3()

        theta = np.radians(30)
        wRc = gtsam.Rot3(np.array([[-math.sin(theta), 0, math.cos(theta)],
                                [-math.cos(theta), 0, -math.sin(theta)],
                                [0, 1, 0]]))

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

    def print(self, d_map):
        for i,point in enumerate(d_map[1]):
            print('Point_',i,':',point)
        for i,pose in enumerate(d_map[0]):
            print('Pose_',i,':',pose)

    def visual_sfm_plot(self,result):

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
        axes.set_xlim3d(-10, 10)
        axes.set_ylim3d(-10, 10)
        axes.set_zlim3d(-10, 10)
        plt.legend()
        plt.show()

    def plot_localization(self, fignum, result):

        fig = plt.figure(fignum)
        axes = fig.gca(projection='3d')
        plt.cla()

        # Plot points
        # Can't use data because self.img_pose[i+1]rent frame might not see all points
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
        plt.pause(60)

if __name__ == "__main__":
    fe = MappingFrontEnd()

    fe.get_image_paths()
    fe.extract_all_image_features()
    fe.feature_matching()
    fe.initial_estimation()

    tic_ba = time.time()    
    sfm_result,nrCamera, nrPoint = fe.bundle_adjustment()
    toc_ba = time.time()
    print('BA spents ',toc_ba-tic_ba,'s')
    print(sfm_result)

    fe.visual_sfm_plot(sfm_result)

    # _, result =fe.sim3_transform_map(sfm_result, nrCamera, nrPoint)
    # fe.print(result)


    # fe.plot_localization(0, result)