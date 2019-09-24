"""Save camera capture images."""
# cSpell: disable
# pylint: disable=no-member, invalid-name
import cv2
from collect_data.scripts.config import video_capture, skip, save

cap = cv2.VideoCapture(video_capture)
count = 0

while True:
    ret, frame = cap.read()
    if ret is False:
        print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
    if count < skip:
        count += 1
        continue
    cv2.imshow('image', frame)

    if save == 1:
        cv2.imwrite("collect_data/videos/4/raw_frame_%d" %
                    count + ".jpg", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destoryAllWindows()
