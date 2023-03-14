import sys
import cv2
import numpy as np
import openpifpaf
from PIL import Image
from typing import List, Tuple
from vector_functions import correct_vectors
from draw_function import draw_vectors, draw_result, draw_vectors_0
from regist_functions import register
from recognition_pose import capture_leader, capture_players
from display_functions import display_registered_playeres
from calculation import compare_pose, calc_multiSimilarity
from settings import SCALE_UP


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

print("Start Recognize...")

#leaderの写真を取得する
leader_picture = capture_leader(capture)

#leaderの姿勢を推定する
leader_pose, gt_anns, meta = predictor.numpy_image(leader_picture)
vectors = correct_vectors(leader_pose, 0)
leader_picture = draw_vectors_0(leader_picture, vectors)

#leaderのスクショを表示
cv2.imshow('Camera 1',leader_picture)
while True:
    # Enterキーを押すまで、スクショを表示
    if cv2.waitKey(10) == 0x0d:
        print('Save frame...')
        break

#プレイヤーを2人としてテスト
plyaer_pictures = capture_players(capture, 3)



capture.release()
cv2.destroyAllWindows()