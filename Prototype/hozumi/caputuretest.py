import sys
import cv2
import numpy as np
import openpifpaf
from PIL import Image
from typing import List, Tuple


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

    # predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
    # predictions, gt_anns, meta = predictor.numpy_image(frame)

    # annotation_painter = openpifpaf.show.AnnotationPainter()
    # with openpifpaf.show.image_canvas(frame) as ax:
    #     annotation_painter.annotations(ax, predictions)
    #     plt.imshow(frame)
    #     plt.show()
    cv2.imshow('Camera 1',frame)

    # ESCキーを押すと終了
    if cv2.waitKey(100) == 0x1b:
        print('ESC pressed. Exiting ...')
        break

capture.release()
cv2.destroyAllWindows()