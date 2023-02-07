import cv2
from typing import Tuple, List
import numpy as np
from settings import SCALE_UP

def get_draw_info(pt1: np.ndarray, pt2: np.ndarray) -> List[Tuple[int, int]]:
    """座標点のxy座標を取得

    Args: 
        pt1 (np.ndarray): 1つ目の点
        pt2 (np.ndarray): 2つ目の点

    Returns:
        List[Tuple[int, int]]: それぞれのxy座標をタプルでまとめたリスト
    """    
    pt1_x = int(pt1[0] * SCALE_UP)
    pt1_y = int(pt1[1] * SCALE_UP)
    pt2_x = int(pt2[0] * SCALE_UP)
    pt2_y = int(pt2[1] * SCALE_UP)
    
    return [(pt1_x, pt1_y), (pt2_x, pt2_y)]



def correct_vectors(landmarks: np.ndarray, index: int) -> List[Tuple[np.ndarray]]:
    """関節点のつながりをまとめた情報をつくる

    Args:
        landmarks (np.ndarray): 複数人のランドマーク情報
        index (int): 何人目のランドマークについて調べるか

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: 関節点のつながり
        [0]:両耳を結んだベクトル
        [1]:左上腕のベクトル
        [2]:左前腕のベクトル
        [3]:右上腕のベクトル
        [4]:右前腕のベクトル
        [5]:両肩を結んだベクトル
        [6]:左大腿部のベクトル
        [7]:左下腿部のベクトル
        [8]:右大腿部のベクトル
        [9]:右下腿部のベクトル
        [10]:腰のベクトル
        [11]:胴体左のベクトル
        [12]:胴体右のベクトル
    """    
    connected: List[Tuple[np.ndarray]] = []

    left_ear = landmarks[index].data[3]
    right_ear = landmarks[index].data[4]
    #両耳を結んだベクトル
    connected.extend([(left_ear, right_ear)])

    left_shoulder = landmarks[index].data[5]
    left_elbow = landmarks[index].data[7]
    left_hand = landmarks[index].data[9]
    #左上腕のベクトル
    connected.extend([(left_elbow, left_shoulder)])
    #左前腕のベクトル
    connected.extend([(left_hand, left_elbow)])

    right_shoulder = landmarks[index].data[6]
    right_elbow = landmarks[index].data[8]
    right_hand = landmarks[index].data[10]
    #右上腕のベクトル
    connected.extend([(right_elbow, right_shoulder)])
    #右前腕のベクトル
    connected.extend([(right_hand, right_elbow)])
    #両肩を結んだベクトル
    connected.extend([(left_shoulder, right_shoulder)])
    
    left_hip = landmarks[index].data[11]
    left_knee = landmarks[index].data[13]
    left_ankle = landmarks[index].data[15]
    #左大腿部のベクトル
    connected.extend([(left_hip, left_knee)])
    #左下腿部のベクトル
    connected.extend([(left_knee, left_ankle)])

    right_hip = landmarks[index].data[12]
    right_knee = landmarks[index].data[14]
    right_ankle = landmarks[index].data[16]
    #右大腿部のベクトル
    connected.extend([(right_hip, right_knee)])
    #右下腿部のベクトル
    connected.extend([(right_knee, right_ankle)])
    #腰のベクトル
    connected.extend([(left_hip, right_hip)])
    #胴体左のベクトル
    connected.extend([(left_shoulder, left_hip)])
    #胴体右のベクトル
    connected.extend([(right_shoulder, right_hip)])

    return connected





