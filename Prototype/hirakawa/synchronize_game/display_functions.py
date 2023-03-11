import cv2
import openpifpaf
import numpy as np
from typing import List, Tuple
from functions import get_draw_info, create_connected
from draw_function import draw_peopleNum
from settings import SCALE_UP, Result_X, Result_Y 
from area_settings import X_LIMIT_START, Y_LIMIT_START, X_LIMIT_END, Y_LIMIT_END, face_width, face_height, Window_width, Window_height

whiteboard = 255 * np.ones([Window_height, Window_width, 3])

def display_registered_playeres(face_Imgs: List[np.array]):
    playeresImg = whiteboard.copy()
    x_offset=int(Window_width/2 - face_width/2)
    y_offset=int(Window_height/2 - face_height/2)

    #テスト用
    #print(x_offset)
    #print(y_offset)
    #print(face_Imgs[0])
    #print(type(face_Imgs[0][0,0,0]))

    face_Img = face_Imgs[0]
    #print(type(face_Img[0,0,0]))
    playeresImg[y_offset:y_offset+face_Img.shape[0], x_offset:x_offset+face_Img.shape[1]] = face_Img.copy()
    #print(playeresImg[y_offset:y_offset+face_Img.shape[0], x_offset:x_offset+face_Img.shape[1]])
    playeresImg = playeresImg.astype('uint8')

    while True:
        cv2.imshow('Camera 1',playeresImg) #認識した顔の画像を表示
                
        # Enterキーを押すと表示終了
        if cv2.waitKey(10) == 0x0d:
            print('Enter pressed. End face display...')
            break

    return playeresImg