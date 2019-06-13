"""
This module test the opencv functions such as findEssentialMat(), recoverPose(), and triangulatePoints().
"""

import unittest
import cv2
import numpy as np
import gtsam 
from matplotlib import pyplot as plt

def create_point_pair(choice):
    src = np.array([], dtype=np.float).reshape(0,2)
    dst = np.array([], dtype=np.float).reshape(0,2)

    if choice == 1:
    # 10 matched key point pairs in a pair of 160x120 distorted images. 
    # Key points are upsampled data with scale 4
        src = np.vstack((src,np.array([548,248])))
        dst = np.vstack((dst,np.array([536,228])))        
            
        src = np.vstack((src,np.array([228,252])))
        dst = np.vstack((dst,np.array([216,236])))
            
        src = np.vstack((src,np.array([184,264])))
        dst = np.vstack((dst,np.array([180,244])))
        
        src = np.vstack((src,np.array([368,440])))
        dst = np.vstack((dst,np.array([360,424])))
        
        src = np.vstack((src,np.array([548,288])))
        dst = np.vstack((dst,np.array([536,268])))
        
        src = np.vstack((src,np.array([328,400])))
        dst = np.vstack((dst,np.array([316,384])))
        
        src = np.vstack((src,np.array([144,244])))
        dst = np.vstack((dst,np.array([136,228])))
        
        src = np.vstack((src,np.array([216,384])))
        dst = np.vstack((dst,np.array([204,368])))
        
        src = np.vstack((src,np.array([212,324])))
        dst = np.vstack((dst,np.array([200,308])))
        
        src = np.vstack((src,np.array([168,368])))
        dst = np.vstack((dst,np.array([156,352])))

    elif choice == 2:
        # 10 matched key point pairs in a pair of 640x480 distorted images
        src = np.vstack((src,np.array([244, 525])))
        dst = np.vstack((dst,np.array([210, 564])))        
            
        src = np.vstack((src,np.array([55, 232])))
        dst = np.vstack((dst,np.array([21, 255])))
            
        src = np.vstack((src,np.array([425, 318])))
        dst = np.vstack((dst,np.array([398, 356])))
        
        src = np.vstack((src,np.array([460, 373])))
        dst = np.vstack((dst,np.array([426, 409])))
        
        src = np.vstack((src,np.array([137, 491])))
        dst = np.vstack((dst,np.array([94, 506])))
        
        src = np.vstack((src,np.array([178, 496])))
        dst = np.vstack((dst,np.array([138, 524])))
        
        src = np.vstack((src,np.array([186, 462])))
        dst = np.vstack((dst,np.array([153, 496])))
        
        src = np.vstack((src,np.array([154, 374])))
        dst = np.vstack((dst,np.array([117, 408])))
        
        src = np.vstack((src,np.array([166, 446])))
        dst = np.vstack((dst,np.array([123, 471])))
        
        src = np.vstack((src,np.array([232, 499])))
        dst = np.vstack((dst,np.array([185, 540])))

    elif choice == 3:
        # 10 matched key point pairs in a pair of 640x480 undistort image pair
        src = np.vstack((src,np.array([245.00, 498.00])))
        dst = np.vstack((dst,np.array([211, 536])))        
            
        src = np.vstack((src,np.array([278, 425])))
        dst = np.vstack((dst,np.array([253, 458])))
            
        src = np.vstack((src,np.array([442, 218])))
        dst = np.vstack((dst,np.array([373, 264])))
        
        src = np.vstack((src,np.array([241, 565])))
        dst = np.vstack((dst,np.array([208, 613])))
        
        src = np.vstack((src,np.array([82, 344])))
        dst = np.vstack((dst,np.array([17, 369])))
        
        src = np.vstack((src,np.array([246, 442])))
        dst = np.vstack((dst,np.array([222, 477])))
        
        src = np.vstack((src,np.array([440, 403])))
        dst = np.vstack((dst,np.array([396, 434])))
        
        src = np.vstack((src,np.array([258, 474])))
        dst = np.vstack((dst,np.array([232, 511])))
        
        src = np.vstack((src,np.array([384, 312])))
        dst = np.vstack((dst,np.array([352, 347])))
        
        src = np.vstack((src,np.array([215, 318])))
        dst = np.vstack((dst,np.array([191, 346])))

    mask = np.array([1,1,1,1,1,1,1,1,1,1])
    return src,dst,mask

class TestFrontEnd(unittest.TestCase):
    def setUp(self):
        # self.calibration = gtsam.Cal3_S2(160, 640, 480).matrix()
        # self.calibration = gtsam.Cal3_S2(fx=156.3, fy=236,s=1,u0=320, v0=240).matrix()
        self.image_transform = np.array([[0.25,0,0.125-0.5],[0,0.25,0.125-0.5],[0,0,1]])
        self.calibration = gtsam.Cal3_S2(fx=333.4, fy=327.22,s=0,u0=303.6, v0=247.6).matrix()
        # self.calibration = gtsam.Cal3_S2(fx=343.55, fy=314.7,s=0,u0=295.97, v0=261.53).matrix()
        # self.calibration = np.dot(self.image_transform,self.calibration2)
        self.distortion = np.array([-0.282548, 0.054412, -0.001882, 0.004796, 0.000000])
        # self.distortion = np.array([-0.305247, 0.064438, -0.007641, 0.006581, 0.000000])
        self.rectification = np.identity(3)
        self.projection = np.array([[226.994629 ,0.000000, 311.982613, 0.000000],[0.000000, 245.874146, 250.410089, 0.000000],[0.000000, 0.000000, 1.000000, 0.000000]])
        # self.projection = np.array([[235.686951, 0.000000, 304.638818, 0.000000],[0.000000, 256.651520, 265.858792, 0.000000],[0.000000, 0.000000, 1.000000, 0.000000]])
        # self.projection = np.dot(self.image_transform,self.projection2)        
        self.calibration_inv = np.linalg.inv(self.calibration)
        self.R = np.array([[1,0,0],[0,1,0],[0,0,1]])
        # self.t = np.array([[-0.79],[-1.36832],[0]])
        self.t = np.array([[1.58],[0],[0]])
        # self.t = np.array([[-0.61],[-0.78832],[0]])
        self.t_skew_symmetric = np.array([[0,-self.t[2],self.t[1]],[self.t[2],0,-self.t[0]],[-self.t[1],self.t[0],0]])
        self.E = np.dot(self.t_skew_symmetric,self.R)
        self.F = np.dot(np.dot(self.calibration_inv.T,self.E),self.calibration_inv)
        # http://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
        # self.P_0 = np.dot(self.calibration,np.hstack((np.identity(3),np.zeros((3,1)))))
        # self.P_1 = np.dot(self.calibration,np.hstack((self.R.T,-np.dot(self.R.T,self.t))))
        self.P_0 = np.hstack((np.identity(3),np.zeros((3,1))))
        self.P_1 = np.hstack((self.R.T,-np.dot(self.R.T,self.t)))
        
    def test_ORB(self):
        img1 = cv2.imread('datasets/4.5/mapping/1_frame.jpg',0)          # queryImage
        img2 = cv2.imread('datasets/4.5/mapping/2_frame.jpg',0) # trainImage

        # Initiate SIFT detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])


        img3=np.array([])
        # cv2.drawMatchesKnn expects list of lists as matches.
        img3=cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img3,flags=2)

        # plt.imshow(img3),plt.show()
        src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ])
        dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ])
        # print(src_pts)
        src = cv2.undistortPoints(np.expand_dims(src_pts, axis=1), cameraMatrix=self.calibration, distCoeffs=self.distortion,R = self.rectification, P=self.projection)
        dst = cv2.undistortPoints(np.expand_dims(dst_pts, axis=1), cameraMatrix=self.calibration, distCoeffs=self.distortion,R = self.rectification, P=self.projection) 
        E, mask = cv2.findEssentialMat(dst,src, cameraMatrix = self.calibration,method =cv2.RANSAC,prob=0.999,threshold=3)
        print("Calculate E with ORB:\n",E)
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    # def test_fundamental_matrix(self):
    #     src,dst,expected_mask = create_point_pair()

    #     F_, inliers= cv2.findFundamentalMat(src,dst,cv2.FM_RANSAC,3,0.99)
    #     E = np.dot(np.dot(self.calibration.T,F_),self.calibration)
    #     actual_E_norm = E/np.linalg.norm(E)  
    #     # np.testing.assert_array_equal(F_,self.F)
    #     # np.testing.assert_array_equal(inliers,expected_mask.reshape(10,1))
    #     print("F to E:",actual_E_norm)


    def test_essential_matrix(self):
        # self.calibration = np.dot(self.image_transform,self.calibration)
        # self.projection = np.dot(self.image_transform,self.projection)
        src,dst,expected_mask = create_point_pair(1)
        # src/=4
        # dst/=4
        # src[:, 0], src[:, 1] = src[:, 1], src[:, 0].copy()
        # dst[:, 0], dst[:, 1] = dst[:, 1], dst[:, 0].copy()
        # print(src)


        # https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
        src = cv2.undistortPoints(np.expand_dims(src, axis=1), cameraMatrix=self.calibration, distCoeffs=self.distortion,R = self.rectification, P=self.projection)
        dst = cv2.undistortPoints(np.expand_dims(dst, axis=1), cameraMatrix=self.calibration, distCoeffs=self.distortion,R = self.rectification, P=self.projection)   

        E, mask = cv2.findEssentialMat(dst,src, cameraMatrix = self.calibration,method =cv2.RANSAC,prob=0.999,threshold=0.1)
        expected_E_norm = self.E/np.linalg.norm(self.E)
        actual_E_norm = E/np.linalg.norm(E) 
        _, local_R, local_t, mask =cv2.recoverPose(E,dst,src,cameraMatrix = self.calibration)
        print("Calculated R with actual E:\n",local_R)
        print("Calculated t with actual E:\n",local_t) 
        # for i,check in enumerate(mask):
        #     if check>0:
        #         s_homo = np.ones((1,3))
        #         d_homo = np.ones((1,3))
        #         s_homo[0][0], s_homo[0][1] = src[i][0][0], src[i][0][1]
        #         d_homo[0][0], d_homo[0][1] = dst[i][0][0], dst[i][0][1]
        #         E = np.array([[0,0,-1.36],[0,0,0.79],[1.36,-0.79,0]])
        #         zero = np.dot(np.dot(d_homo,E),s_homo.T)
        #         self.assertAlmostEqual(zero,0,delta=0.05)
        print('Expected E:',expected_E_norm)
        # print('Actual E:',actual_E_norm)
        # self.assertAlmostEqual(self.E,E)

    # def test_pose_recover_matrix(self):
    #     src,dst,expected_mask = create_point_pair() 
    #     E = np.array([[0,0,-1.36],[0,0,0.79],[1.36,-0.79,0]])
    #     _, local_R, local_t, mask =cv2.recoverPose(E,dst,src,cameraMatrix = self.calibration)
    #     print("R:",local_R)
    #     print("t:",local_t)

    def test_triangulation(self):
        np.set_printoptions(suppress=True,
            formatter={'float_kind':'{:0.2f}'.format})
        src,dst,expected_mask = create_point_pair(3)
        src0 = cv2.undistortPoints(np.expand_dims(src, axis=1), cameraMatrix=self.calibration, distCoeffs=None)
        dst0 = cv2.undistortPoints(np.expand_dims(dst, axis=1), cameraMatrix=self.calibration, distCoeffs=None)         
        src = np.expand_dims(src, axis=1)
        dst = np.expand_dims(dst, axis=1)
        # F_, mask= cv2.findFundamentalMat(src,dst,cv2.FM_RANSAC,0.004,0.99)

        E, mask = cv2.findEssentialMat(dst,src, cameraMatrix = self.calibration,method =cv2.RANSAC,prob=0.999,threshold=0.004)
        actual_E_norm = E/np.linalg.norm(E) 
        print('Actual E:',actual_E_norm)
        print(mask)
        _, local_R, local_t, mask =cv2.recoverPose(E,dst,src,cameraMatrix = self.calibration)
        print("R:",local_R)
        print("t:",local_t)
        s = 1.38/local_t[0]
        local_t[0] *= s
        local_t[2] *= s 
        print("t:",local_t)
        # self.P_0 = np.dot(self.calibration,np.hstack((np.identity(3),np.zeros((3,1)))))
        # P = np.dot(self.calibration,np.hstack((local_R.T,-np.dot(local_R.T,local_t))))
        P = np.hstack((local_R.T,-np.dot(local_R.T,local_t)))
        # points4D = cv2.triangulatePoints(self.P_0,P,src, dst)
        points4D = cv2.triangulatePoints(self.P_0,P,src0, dst0)
        points4D = points4D / np.tile(points4D[-1, :], (4, 1))
        pt3d = points4D
        
        # points4D = points4D / np.tile(points4D[-1, :], (4, 1))
        # pt3d = points4D[:3, :].T
        # for i,good in enumerate(mask):
        #     if good ==255:
        #         print("Recover Points:\n",pt3d[:3].T[i])

        s = np.dot(self.P_0,points4D)
        s/=s[2]
        d=np.dot(P,points4D)
        d/=d[2]
        for i,good in enumerate(mask):
            if good==255:
                print("pt3d",pt3d.T[i])
                print("src",src0[i,0,:])
                print("src",s.T[i])
                print("dst",dst0[i,0,:])
                print("dst",d.T[i])

        # s = np.dot(self.P_0,points4D)
        # s/=s[2]
        # d=np.dot(self.P_1,points4D)
        # d/=d[2]
        # for i,good in enumerate(mask):
        #     if good==255:
        #         print("pt3d",pt3d.T[i])
        #         print("src",src[i,:])
        #         print("src",s.T[i])
        #         print("dst",dst[i,:])
        #         print("dst",d.T[i])



if __name__ == "__main__":
    unittest.main()