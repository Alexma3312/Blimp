# cSpell: disable
# pylint: disable=no-member, invalid-name
import cv2

cap = cv2.VideoCapture(1)
count = 0
img_idx = 0

while True:
    ret, frame = cap.read()
    cv2.imshow('image', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        if count > 7:
            count = 0
            img_idx += 1
        if count < 0:
            count = 0
        cv2.imwrite("raw_frame_%d" % img_idx+"_%d.jpg" % count, frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        count += 1
    if cv2.waitKey(1) & 0xFF == ord('m'):
        count += 1
    if cv2.waitKey(1) & 0xFF == ord('n'):
        count -= 2
    if cv2.waitKey(1) & 0xFF == ord('k'):
        img_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('l'):
        img_idx -= 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break
cap.release()
cv2.destoryAllWindows()
