import sys
import cv2
import numpy as np
import openpifpaf
from PIL import Image
from typing import List, Tuple
from functions import draw_line, create_connected
from draw_function import draw_landmarks, draw_rectangle
from settings import SCALE_UP

"""
def draw_landmarks(image: np.ndarray, landmarks: List) -> np.ndarray:

    annotated_image = image.copy()
    # ランドマークとして検出されている点を囲む矩形を描画する
    # body_rectangle: List[float] = landmarks[0].json_data()["bbox"]
    # base_x, base_y, width, height = body_rectangle

    # x1 = int(base_x)
    # y1 = int(base_y - 10)
    # x2 = int(base_x+width)
    # y2 = int(base_x+height)
    # # 解像度1/4にしたので、4倍して位置を調整
    # cv2.rectangle(annotated_image, (x1*SCALE,y1*SCALE), (x2*SCALE, y2*SCALE), (255, 255, 255))

    connected_keypoints = create_connected(landmarks, index=0)
    for (pt1, pt2) in connected_keypoints:
        if((0 in pt1) or (0 in pt2)): continue # 座標をうまく取得できなかったとき

        annotated_image = draw_line(annotated_image, pt1, pt2)
    
    return annotated_image
"""


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
    read_video: Tuple[bool, np.ndarray] = capture.read()
    success, frame = read_video
    # print("frame1 =",frame)

    if not success :
        print( "frame is None" )
        break
    
    
    resize_frame: np.ndarray = cv2.resize(frame, dsize=None, fx=(1.0 / SCALE_UP), fy=(1.0 / SCALE_UP))
    predictions, gt_anns, meta = predictor.numpy_image(resize_frame)
    """
    predictions：関節座標
    インデックス：関節座標点
    """
    if len(predictions) == 0: continue

    #骨格を表示
    #annotated_image: np.ndarray = draw_landmarks(frame, predictions)
    #predictions[0].data[0] : (x,y,c)
    
    #外接矩形を表示
    annotated_image: np.ndarray = draw_rectangle(frame, predictions)

    height = frame.shape[0]
    width = frame.shape[1]
    annotated_image = cv2.flip(annotated_image, 1)
    bigger_frame = cv2.resize(annotated_image, (int(width) * 2, int(height) * 2))
    cv2.imshow('Camera 1',bigger_frame)
    #cv2.moveWindow("Camera 1", 200,40)

    # ESCキーを押すと終了
    if cv2.waitKey(100) == 0x1b:
        print('ESC pressed. Exiting ...')
        break

capture.release()
cv2.destroyAllWindows()