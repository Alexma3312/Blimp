# cSpell: disable
# pylint: disable=no-member, invalid-name
import cv2

from utilities.video_streamer import VideoStreamer

if __name__ == '__main__':
    basedir = "camera"
    # basedir = "datasets/videos/1/"
    camid = 0
    height = 480
    width = 640
    skip = 0
    start_index = 1
    img_glob = "*.jpg"
    vs = VideoStreamer(basedir, camid, height, width, skip, img_glob, start_index)
    while True:
        # Get a new image.
        img, status = vs.next_frame()
        if status is False:
            break
        if img is None:
            continue
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destoryAllWindows()
