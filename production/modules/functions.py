import cv2
from typing import Tuple, List
import numpy as np
from .settings import SCALE_UP
from .area_settings import X_LIMIT_START, Y_LIMIT_START

def get_draw_info(pt1: np.ndarray, pt2: np.ndarray) -> List[Tuple[int, int]]:
    """座標点のxy座標を取得

    Args: 
        pt1 (np.ndarray): 1つ目の点
        pt2 (np.ndarray): 2つ目の点

    Returns:
        List[Tuple[int, int]]: それぞれのxy座標をタプルでまとめたリスト
    """    
    
    pt1_x = int(pt1[0] * SCALE_UP + X_LIMIT_START)
    pt1_y = int(pt1[1] * SCALE_UP + Y_LIMIT_START)
    pt2_x = int(pt2[0] * SCALE_UP + X_LIMIT_START)
    pt2_y = int(pt2[1] * SCALE_UP + Y_LIMIT_START)
    
    return [(pt1_x, pt1_y), (pt2_x, pt2_y)]

def get_draw_info_0(pt1: np.ndarray, pt2: np.ndarray, scale:float = 1) -> List[Tuple[int, int]]:
    """座標点のxy座標を取得(全画面版)

    Args: 
        pt1 (np.ndarray): 1つ目の点
        pt2 (np.ndarray): 2つ目の点

    Returns:
        List[Tuple[int, int]]: それぞれのxy座標をタプルでまとめたリスト
    """    
    
    pt1_x = int(pt1[0] * scale)
    pt1_y = int(pt1[1] * scale)
    pt2_x = int(pt2[0] * scale)
    pt2_y = int(pt2[1] * scale)
    
    return [(pt1_x, pt1_y), (pt2_x, pt2_y)]


def create_connected(landmarks: np.ndarray, index: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """関節点のつながりをまとめた情報をつくる

    Args:
        landmarks (np.ndarray): 複数人のランドマーク情報
        index (int): 何人目のランドマークについて調べるか

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: 関節点のつながり
    """    
    connected: List[Tuple[np.ndarray, np.ndarray]] = []

    left_shoulder = landmarks[index].data[5]
    left_elbow = landmarks[index].data[7]
    left_hand = landmarks[index].data[9]
    connected.extend([(left_hand, left_elbow), (left_elbow, left_shoulder)])

    right_shoulder = landmarks[index].data[6]
    right_elbow = landmarks[index].data[8]
    right_hand = landmarks[index].data[10]
    connected.extend([(right_hand, right_elbow), (right_elbow, right_shoulder),
                      (left_shoulder, right_shoulder)])
    
    left_hip = landmarks[index].data[11]
    left_knee = landmarks[index].data[13]
    left_ankle = landmarks[index].data[15]
    connected.extend([(left_hip, left_knee), (left_knee, left_ankle)])

    right_hip = landmarks[index].data[12]
    right_knee = landmarks[index].data[14]
    right_ankle = landmarks[index].data[16]
    connected.extend([(right_hip, right_knee), (right_knee, right_ankle),
                      (left_hip, right_hip),(left_shoulder, left_hip),
                      (right_shoulder, right_hip)])
    

    return connected





