import sys
import cv2
import numpy as np


# PCに繋がっているUSBカメラから撮る場合はこれ
capture = cv2.VideoCapture(0)

# AXISのカメラから映像を撮る場合はこれ
# capture = cv2.VideoCapture('rtsp://user:password@192.168.1.1/axis-media/media.amp')
if not capture.isOpened():
    print( "Error opening capture device")
    capture.release()
    cv2.destroyAllWindows()

if capture.isOpened():
    print( "Device captured correctly",capture)

while 1:

 ret, frame = capture.read()
 # print("frame1 =",frame)

 if ret == False :
  print( "frame is None" )
  break

 cv2.imshow('Camera 1',frame)

 # ESCキーを押すと終了
 if cv2.waitKey(100) == 0x1b:
  print('ESC pressed. Exiting ...')
  break

capture.release()
cv2.destroyAllWindows()