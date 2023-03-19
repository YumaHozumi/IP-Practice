import sys
import cv2
import numpy as np
import openpifpaf
from PIL import Image
from typing import List, Tuple
from vector_functions import correct_vectors
from draw_function import draw_vectors, draw_result, draw_vectors_0
from regist_functions import register
from display_functions import display_registered_playeres, display_result, display_change, display_final_result
from recognition_pose import get_humanPicture
from calculation import compare_pose, calc_multiSimilarity
from settings import SCALE_UP
from area_settings import Window_width, Window_height


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
        bigger_frame = cv2.resize(annotated_image, (Window_width, Window_height))
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
        annotated_image = draw_vectors_0(annotated_image, vectors, SCALE_UP)
        people_vectors[person_id] = np.asarray(vectors)

    height = frame.shape[0]
    width = frame.shape[1]
    annotated_image = cv2.flip(annotated_image, 1)

    bigger_frame = cv2.resize(annotated_image, (Window_width, Window_height))
    cv2.imshow('Camera 1',bigger_frame)
    #cv2.moveWindow("Camera 1", 200,40)

    
    if cv2.waitKey(10) == 0x0d: 
        # Enterキーを押すとプレイヤー登録開始
        face_Imgs = register(capture, predictor)

        playerNum = len(face_Imgs) #プレイ人数
        #全ての類似度を格納する
        all_similarities = []
        for all_people in range(playerNum):
            all_similarities.append([])
        #ここまででゲームの前段階終了

        for leader_id in range(playerNum):
            #playersのid(インデックス)を格納したリストを作成する
            players_id:List = []
            for i in range(playerNum):
                if not i == leader_id:
                    players_id.append(i)

            leader_picture, leader_vectors, player_pictures, players_vectors = get_humanPicture(capture, predictor, face_Imgs, players_id, leader_id)

            #ここまでで画像の抽出は完了

            #以降で、類似度の計算・結果の表示
            similarities = []
            for j in range(len(players_vectors)):
                similarity = compare_pose(leader_vectors, players_vectors[j]) * 100
                similarities.append(similarity)
                all_similarities[players_id[j]].append(similarity)
            
            display_result(player_pictures, leader_id, players_id, similarities)
            while True:
                # Enterキーを押すと次へ
                if cv2.waitKey(10) == 0x0d:
                    break
            
            #最後の1回以外は、交代のメッセージを表示する
            if not (leader_id == (playerNum - 1)):
                display_change()
                while True:
                    # Enterキーを押すと次へ
                    if cv2.waitKey(10) == 0x0d:
                        break
        break

display_final_result(face_Imgs, all_similarities)
while True:
    # Enterキーを押すと終了
    if cv2.waitKey(10) == 0x0d:
        break
print(all_similarities)
#print(type(face_Imgs))
capture.release()
cv2.destroyAllWindows()