import sys
import cv2
import numpy as np
import openpifpaf
from PIL import Image
from typing import List, Tuple
from modules.vector_functions import correct_vectors
from modules.draw_function import draw_vectors_0, draw_vectors, draw_result
from modules.regist_functions import register
from modules.display_functions import display_title, display_rule, display_gameNum, display_finalMessage, cv2_putText
from modules.display_functions import display_registered_playeres, display_result, display_change, display_final_result
from modules.recognition_pose import get_humanPicture
from modules.calculation import compare_pose, calc_multiSimilarity, compare_pose_toGame
from modules.settings import SCALE_UP, Capture
from modules.area_settings import Window_width, Window_height
import tkinter as tk
import time

def countdown(n, label):
    while n >= 0:
        label['text'] = n
        n -= 1
        time.sleep(1)
    label['text'] = 'Done!'


if __name__ == '__main__':
    # PCに繋がっているUSBカメラから撮る場合はこれ(settings.pyから取得)
    capture = Capture

    #ウインドウの設定
    cv2.namedWindow('Camera 1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera 1', Window_width, Window_height) 

    if not capture.isOpened(): # 正常に動画読み込めなかったとき
        print( "Error opening capture device")
        capture.release() # カメラデバイス閉じる
        cv2.destroyAllWindows() # 開いているすべてのウィンドウ閉じる
        exit()

    if capture.isOpened(): # 正常に読みこめたとき
        print( "Device captured correctly",capture)

    predictor = openpifpaf.Predictor(checkpoint = "shufflenetv2k16")

    print("Start System...")

    #待機画面で表示するテキスト
    over_text = "プレイヤー待機中..."
    under_text = "Press Enter To Start"


    while capture.isOpened():
        read_video: Tuple[bool, np.ndarray] = capture.read()
        success, frame = read_video
        # print("frame1 =",frame)

        if not success :
            print( "frame is None" )
            break
        
        #待機画面では解像度を1/8に落として骨格推定と表示を行う
        resize_frame: np.ndarray = cv2.resize(frame, dsize=None, fx=(1.0 / 8), fy=(1.0 / 8))
        predictions, gt_anns, meta = predictor.numpy_image(resize_frame)
        
        annotated_image = frame.copy()

        #認識領域に人が映ってないときにもカメラ映像を出すように
        if len(predictions) == 0: 
            annotated_image = cv2.flip(annotated_image, 1)
            display_frame = cv2.resize(annotated_image, (Window_width, Window_height))
            display_frame = cv2_putText(display_frame, over_text, (int(Window_width/2), 80), 100, (0,255,0), 2)
            display_frame = cv2_putText(display_frame, under_text, (int(Window_width/2), Window_height - 100), 100, (0,255,0), 2)
            cv2.imshow('Camera 1',display_frame)
            
            # ESCキーを押すと終了
            if cv2.waitKey(10) == 0x1b:
                print('ESC pressed. Exiting ...')
                break
            
            continue

        #骨格(ベクトル)を表示
        people_vectors: np.ndarray = np.zeros((len(predictions), 13, 2, 3))
        
        for person_id in range(len(predictions)):
            vectors = correct_vectors(predictions, person_id)
            annotated_image = draw_vectors_0(annotated_image, vectors, 8) #骨格の表示
            people_vectors[person_id] = np.asarray(vectors)

        annotated_image = cv2.flip(annotated_image, 1)
        display_frame = cv2.resize(annotated_image, (Window_width, Window_height))
        display_frame = cv2_putText(display_frame, over_text, (int(Window_width/2), 80), 100, (0,255,0), 2)
        display_frame = cv2_putText(display_frame, under_text, (int(Window_width/2), Window_height - 100), 100, (0,255,0), 2)
        cv2.imshow('Camera 1',display_frame)
        #cv2.moveWindow("Camera 1", 200,40)


        if cv2.waitKey(10) == 0x0d: 
            #タイトルを表示
            display_title()

            # Enterキーを押すとプレイヤー登録開始
            face_Imgs, display_face_Imgs = register(capture, predictor)

            playerNum = len(face_Imgs) #プレイ人数
            #全ての類似度を格納する
            all_similarities: List[list] = [[] for _ in range(playerNum)]
            #ここまででゲームの前段階終了

            #各役割の概要を説明
            display_rule()

            #ゲーム開始
            for leader_id in range(playerNum):
                #何ゲーム目かを表示
                display_gameNum(leader_id)

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
                    similarity = compare_pose_toGame(leader_vectors, players_vectors[j]) * 100
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

            #総合結果を表示
            display_final_result(display_face_Imgs, all_similarities)
            while True:
                # Enterキーを押すと終了
                if cv2.waitKey(10) == 0x0d:
                    break
            #最後のメッセージを表示
            display_finalMessage()

    #ラグを減らすために、プログラムはCtrl+Cでのみ止まるように        
    #print(type(face_Imgs))
    capture.release()
    cv2.destroyAllWindows()