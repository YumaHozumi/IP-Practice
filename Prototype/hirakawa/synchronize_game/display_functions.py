import cv2
import openpifpaf
import numpy as np
from typing import List, Tuple
from functions import get_draw_info, create_connected
from draw_function import draw_peopleNum
from settings import SCALE_UP, Result_X, Result_Y 
from area_settings import X_LIMIT_START, Y_LIMIT_START, X_LIMIT_END, Y_LIMIT_END, face_width, face_height, Window_width, Window_height

whiteboard = 255 * np.ones([Window_height, Window_width, 3])

#プレイヤーを識別する色(B,G,R)
player_color: List = [(0, 0, 255),(255, 0, 0),(0, 255, 0),(0, 165, 255)]

def display_registered_playeres(face_Imgs: List[np.array]):
    playeresImg = whiteboard.copy() #背景の設定

    #画面の説明の表示
    cv2.putText(playeresImg, 'Registered Players', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,0), 4, cv2.LINE_AA)
    
    #登録結果表示画面の作成
    people_num = len(face_Imgs)
    for i in range(people_num):
        img = face_Imgs[i] #顔画像

        #描画領域の指定
        separate_width = Window_width / (people_num + 1)
        x_offset=int((i+1)*separate_width - face_width/2)
        y_offset=int(Window_height/2 - face_height/2)
        playeresImg[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img.copy()
        txt = "Player" + str(int(i + 1))
        cv2.putText(playeresImg, txt, (x_offset, y_offset - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.75, player_color[i], 3, cv2.LINE_AA)
        
    #型変換
    playeresImg = playeresImg.astype('uint8')

    while True:
        cv2.imshow('Camera 1',playeresImg) #認識した顔の画像を表示
                
        # Enterキーを押すと表示終了
        if cv2.waitKey(10) == 0x0d:
            print('Enter pressed. End face display...')
            break

    return playeresImg