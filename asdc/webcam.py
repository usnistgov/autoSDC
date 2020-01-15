
import time
import cv2, platform
#import numpy as np

# https://stackoverflow.com/questions/26691189/how-to-capture-video-stream-with-opencv-python

cap = cv2.VideoCapture(2)
time.sleep(1)
if not cap:
    print("!!! Failed VideoCapture: invalid parameter!")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    h, w, c = frame.shape

    if type(frame) == type(None):
        print("!!! Couldn't read frame!")
        break

    frame = cv2.drawMarker(frame, (w//2,h//2), (0,0,0), markerSize=800, thickness=2)	
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the capture
cap.release()
cv2.destroyAllWindows()

