import sys
import cv2
import numpy as np
import openpifpaf
from PIL import Image

# PCに繋がっているUSBカメラから撮る場合はこれ
capture = cv2.VideoCapture(0)

if not capture.isOpened(): # 正常に動画読み込めなかったとき
    print( "Error opening capture device")
    capture.release() # カメラデバイス閉じる
    cv2.destroyAllWindows() # 開いているすべてのウィンドウ閉じる
    exit()

if capture.isOpened(): # 正常に読みこめたとき
    print( "Device captured correctly",capture)

predictor = openpifpaf.Predictor(checkpoint = "shufflenetv2k16")

while capture.isOpened():
    """
    success：画像の取得が成功したか
    frame：RGBの値を持っている3次元の配列データ ex) サイズ (480, 640, 3) 高さ、幅、色チャネル
    """
    success, frame = capture.read()
    # print("frame1 =",frame)

    if not success :
        print( "frame is None" )
        break
    frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    predictions, gt_anns, meta = predictor.numpy_image(frame)

    with openpifpaf.show.image_canvas(frame) as ax:
        height = frame.shape[0]
        width = frame.shape[1]
        bigger_frame = cv2.resize(frame, (int(width*2.0), int(height*2.0)))
        cv2.imshow('Camera 1',bigger_frame)


    # ESCキーを押すと終了
    if cv2.waitKey(100) == 0x1b:
        print('ESC pressed. Exiting ...')
        break

capture.release()
cv2.destroyAllWindows()