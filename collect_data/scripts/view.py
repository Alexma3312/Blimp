"""View web camera input."""
# cSpell: disable
# pylint: disable=no-member, invalid-name
#!/usr/bin/env python
import cv2
from collect_data.scripts.config import video_capture

cap = cv2.VideoCapture(video_capture)
count = 0


while True:

    ret, frame = cap.read()

    cv2.imshow('image', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count += 1

cap.release()
cv2.destoryAllWindows()
