import sys
import cv2
import numpy as np
import openpifpaf
from PIL import Image
from typing import List, Tuple
from vector_functions import correct_vectors
from draw_function import draw_vectors, draw_result, draw_vectors_0
from regist_functions import register
from recognition_pose import capture_leader, extract_leaderArea, capture_players, extract_playersArea
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

#leaderのスクショを取得する
leader_screen = capture_leader(capture)
#対象領域のみ切り取る
leader_picture = extract_leaderArea(leader_screen)

#leaderの姿勢を推定する
leader_pose, gt_anns, meta = predictor.numpy_image(leader_picture)
leader_vectors = correct_vectors(leader_pose, 0)
leader_picture = draw_vectors_0(leader_picture, leader_vectors)

#leaderのスクショを表示
cv2.imshow('Camera 1',leader_picture)
while True:
    # Enterキーを押すまで、スクショを表示
    if cv2.waitKey(10) == 0x0d:
        print('Save frame...')
        break

#playerの姿勢を推定する
playerNum = 1
player_screen = capture_players(capture, playerNum)
player_pictures = extract_playersArea(player_screen, playerNum)
players_vectors = []
for i in range(playerNum):
    player_pose, gt_anns, meta = predictor.numpy_image(player_pictures[i])
    if len(player_pose) > 0:
        players_vectors.append(correct_vectors(player_pose, 0))
    else:
        players_vectors = []
        break
    player_pictures[i] = draw_vectors_0(player_pictures[i], players_vectors[i])

if players_vectors:
    players_result = player_pictures[0]
    for j in range(len(players_vectors) - 1):
        players_result = cv2.hconcat([players_result, player_pictures[j + 1]]) 

    #playerのスクショを表示
    cv2.imshow('Camera 1',players_result)
    while True:
        # Enterキーを押すまで、スクショを表示
        if cv2.waitKey(10) == 0x0d:
            print('Save frame...')
            break


capture.release()
cv2.destroyAllWindows()