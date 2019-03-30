#!/usr/bin/env python
import cv2

cap = cv2.VideoCapture(1)
count = 0


while(True):
  
    ret, frame = cap.read()

    cv2.imshow('image',frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count += 1

cap.release()
cv2.destoryAllWindows()
