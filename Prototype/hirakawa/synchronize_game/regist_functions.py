import cv2
import numpy as np
from typing import List, Tuple
from functions import get_draw_info, create_connected
from settings import SCALE_UP, Result_X, Result_Y

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

def regist_faceImg(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """顔画像を登録する

    Args:
        image (np.ndarray): 登録領域の画像
        landmarks (np.ndarray): 関節点座標のリスト

    Returns:
        np.ndarray: 顔画像のリスト
    """
    
    #関数作成途中
    return landmarks
