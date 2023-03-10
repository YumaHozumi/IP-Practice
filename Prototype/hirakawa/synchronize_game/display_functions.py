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
    prayeresImg = whiteboard.copy()
    x_offset=int(Window_width/2 - face_width/2)
    y_offset=int(Window_height/2 - face_height/2)
    print(x_offset)
    print(y_offset)
    print(face_Imgs[0])
    #prayeresImg[y_offset:y_offset+face_Imgs[0].shape[0], x_offset:x_offset+face_Imgs[0].shape[1], : ] = 0
    prayeresImg[y_offset:y_offset+face_Imgs[0].shape[0], x_offset:x_offset+face_Imgs[0].shape[1]] = face_Imgs[0].copy()
    print(prayeresImg[y_offset:y_offset+face_Imgs[0].shape[0], x_offset:x_offset+face_Imgs[0].shape[1]])

    print(prayeresImg.shape)

    while True:
        cv2.imshow('Camera 1',prayeresImg) #認識した顔の画像を表示
                
        # Enterキーを押すと表示終了
        if cv2.waitKey(10) == 0x0d:
            print('Enter pressed. End face display...')
            break

    return prayeresImg