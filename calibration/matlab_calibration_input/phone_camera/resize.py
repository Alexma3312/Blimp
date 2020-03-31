import cv2
import os
import glob

def resize():
    """A function to undistort the distorted images in a folder."""
    img_paths = glob.glob('/*.jpg')
    img_paths.sort()
    print("Number of Images: ", len(img_paths))
    maxlen = len(img_paths)
    if maxlen == 0:
        raise IOError(
            'No images were found (maybe wrong \'image extension\' parameter?)')

    for img_idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path, 1)
        resize_img = cv2.resize(
            img, (640, 480))

        output_path ='/resize/'+'frame'+'_%d' % img_idx+'.jpg'
        print(output_path)
        cv2.imwrite(output_path, resize_img)

resize()
