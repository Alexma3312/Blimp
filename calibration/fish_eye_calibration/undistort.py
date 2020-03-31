import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob

# You should replace these 3 lines with the output in calibration step
DIM=(640, 480)
K=np.array([[334.8126429037114, 0.0, 328.42985433174505], [0.0, 314.31399085262495, 250.76511423101277], [0.0, 0.0, 1.0]])
D=np.array([[-0.08042315181689647], [0.06882233115527008], [-0.1504221482483509], [0.11002335880674598]])
def undistort(idx, img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite("frame_{}.jpg".format(idx), undistorted_img)
    # cv2.imshow("undistorted", undistorted_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
if __name__ == '__main__':
    images = glob.glob('*.jpg')
    for i,fname in enumerate(images):
        print(fname)
        undistort(i,fname)