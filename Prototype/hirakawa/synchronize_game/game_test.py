import sys
import cv2
import numpy as np
import openpifpaf
from PIL import Image
from typing import List, Tuple
from vector_functions import correct_vectors
from draw_function import draw_vectors, draw_result
from regist_functions import register
from display_functions import display_registered_playeres
from recognition_pose import get_humanPicture
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

print("Start System...")


while capture.isOpened():
    read_video: Tuple[bool, np.ndarray] = capture.read()
    success, frame = read_video
    # print("frame1 =",frame)

    if not success :
        print( "frame is None" )
        break
    
    
    resize_frame: np.ndarray = cv2.resize(frame, dsize=None, fx=(1.0 / SCALE_UP), fy=(1.0 / SCALE_UP))
    predictions, gt_anns, meta = predictor.numpy_image(resize_frame)
    
    annotated_image = frame.copy()

    #認識領域に人が映ってないときにもカメラ映像を出すように
    if len(predictions) == 0: 
        height = frame.shape[0]
        width = frame.shape[1]
        #サイズ確認用(テスト)
        #print('({0}, {1})'.format(height, width))
        
        annotated_image = cv2.flip(annotated_image, 1)
        bigger_frame = cv2.resize(annotated_image, (int(width) * 2, int(height) * 2))
        cv2.imshow('Camera 1',bigger_frame)
        
        # ESCキーを押すと終了
        if cv2.waitKey(10) == 0x1b:
            print('ESC pressed. Exiting ...')
            break
        
        continue

    #骨格(ベクトル)を表示
    people_vectors: np.ndarray = np.zeros((len(predictions), 13, 2, 3))
    
    for person_id in range(len(predictions)):
        vectors = correct_vectors(predictions, person_id)
        annotated_image = draw_vectors(annotated_image, vectors)
        people_vectors[person_id] = np.asarray(vectors)

    height = frame.shape[0]
    width = frame.shape[1]
    annotated_image = cv2.flip(annotated_image, 1)

    bigger_frame = cv2.resize(annotated_image, (int(width) * 2, int(height) * 2))
    cv2.imshow('Camera 1',bigger_frame)
    #cv2.moveWindow("Camera 1", 200,40)

    
    if cv2.waitKey(10) == 0x0d: 
        # Enterキーを押すとプレイヤー登録開始
        print('Start register...')
        face_Imgs = register(capture, predictor)
        print('End register...')

        playerNum = len(face_Imgs)
        print(playerNum)
        #ここまででゲームの前段階終了

        print('Start recognize')
        leader_picture, leader_vectors, player_pictures, players_vectors = get_humanPicture(capture, predictor, playerNum, 0)

        for player_vector in players_vectors:
            similarity = compare_pose(leader_vectors, player_vector)
            print(similarity)
        break


#print(type(face_Imgs))
capture.release()
cv2.destroyAllWindows()