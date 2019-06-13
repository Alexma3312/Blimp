import cv2
import glob
import os
import numpy as np
import cv2
import gtsam

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
    
    # calibration = gtsam.Cal3_S2(fx=331.0165, fy=310.4791,s=0,u0=332.7372, v0=248.5307).matrix()
    # distortion = np.array([-0.3507, 0.1112, 8.6304e-04, -0.0018, 0.000000])
    # rectification = np.identity(3)
    
    # calibration = gtsam.Cal3_S2(fx=332.8312, fy=312.3082,s=-1.1423,u0=330.2353, v0=249.5842).matrix()
    # distortion = np.array([-0.3718, 0.1623, 5.4112e-04, -8.9232e-04, -0.0362])
    # rectification = np.identity(3)

    count = 0
    img_idx = 0
    for i,img_path in enumerate(img_paths):
        img = cv2.imread(img_path, 1)

        # h,  w = img.shape[:2]
        # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(calibration,distortion,(w,h),1,(w,h))
        dst = cv2.undistort(img, calibration, distortion, np.eye(3),projection)
        # dst = cv2.undistort(img, calibration, distortion)

        output_path = 'frame_%d.jpg'%count
        output_path = os.path.join(basedir+'ros_undistort/', output_path)
        cv2.imwrite(output_path,dst)
        count += 1

if __name__ == "__main__":
    undistort_image('camera_calibration/36/','*.jpg')