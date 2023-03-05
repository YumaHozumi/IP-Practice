import cv2
import numpy as np
from typing import List, Tuple
from functions import get_draw_info, create_connected
from settings import SCALE_UP, Result_X, Result_Y 
from area_settings import X_LIMIT_START, Y_LIMIT_START, X_LIMIT_END, Y_LIMIT_END

def registerable_check(landmarks: np.ndarray) -> np.ndarray:
    """顔画像登録が可能な人の関節点情報のみ抽出する

    Args:
        landmarks (np.ndarray): 関節点座標のリスト

    Returns:
        np.ndarray: 登録可能な人を表すラベル
    """

    registable_list = np.ones(len(landmarks)) #登録可能な人を表すラベル

    #登録可能リストを作成する
    for num in range(len(landmarks)):
        parts = landmarks[num]
        #左目と左耳のいずれかが存在し、かつ右目と右耳のいずれか存在する場合登録可能とする
        if(((parts.data[1][2]==0)and(parts.data[3][2]==0)) or ((parts.data[2][2]==0)and(parts.data[4][2]==0))):
            registable_list[num] *= 0
    
    print(registable_list)
    print("--------------")
    return registable_list

def regist_faceImg(register_frame: np.ndarray, landmarks: np.ndarray, label: np.ndarray) -> np.ndarray:
    """顔画像を登録する

    Args:
        register_frame (np.ndarray): 登録領域の画像
        landmarks (np.ndarray): 関節点座標のリスト
        label (np.ndarray): 顔画像を登録できるかのラベル

    Returns:
        np.ndarray: 顔画像のリスト
    """
    
    parts = landmarks[0]

    #顔画像領域を計算
    start_X = 0
    start_Y = 0
    end_X = 0
    end_Y = 0
    if(label[0] == 1):
        if(parts.data[4][2]>0):
            start_X = parts.data[4][0] * SCALE_UP - 20 
            start_Y = parts.data[4][1] * SCALE_UP - 100
            end_Y =  parts.data[4][1] * SCALE_UP + 100
        elif(parts.data[2][2]>0):
            start_X = parts.data[2][0] * SCALE_UP - 60 
            start_Y = parts.data[2][1] * SCALE_UP - 100
            end_Y =  parts.data[2][1] * SCALE_UP + 100

        if(parts.data[3][2]>0):
            end_X = parts.data[3][0] * SCALE_UP + 20
        elif(parts[1][2]>0):
            end_X = parts.data[3][0] * SCALE_UP + 60

        #領域外になった時の処理(正しいかどうかはちょっと・・・)
        if(start_X < 0): start_X = 0
        if(start_Y < 0): start_Y = 0
        if(end_X > (X_LIMIT_END - X_LIMIT_START)): end_X = X_LIMIT_END - X_LIMIT_START
        if(end_Y > (Y_LIMIT_END - Y_LIMIT_START)): end_Y = Y_LIMIT_END - Y_LIMIT_START
        

    #顔画像領域を抽出
    face_frame = register_frame[int(start_Y):int(end_Y), int(start_X):int(end_X)]

    return face_frame
