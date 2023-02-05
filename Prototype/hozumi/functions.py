import cv2
from typing import Tuple, List
import numpy as np


def get_draw_info(pt1: np.ndarray, pt2: np.ndarray) -> List[Tuple[int, int]]:
    """座標点のxy座標を取得

    Args: 
        pt1 (np.ndarray): 1つ目の点
        pt2 (np.ndarray): 2つ目の点

    Returns:
        List[Tuple[int, int]]: それぞれのxy座標をタプルでまとめたリスト
    """    
    SCALE_UP: int = 4
    pt1_x = int(pt1[0] * SCALE_UP)
    pt1_y = int(pt1[1] * SCALE_UP)
    pt2_x = int(pt2[0] * SCALE_UP)
    pt2_y = int(pt2[1] * SCALE_UP)
    
    return [(pt1_x, pt1_y), (pt2_x, pt2_y)]


def draw_line(image: np.ndarray, pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
    """2つの点を線で結ぶ

    Args:
        image (np.ndarray): 点を描画する画像
        pt1 (np.ndarray): 1つ目の点
        pt2 (np.ndarray): 2つ目の点

    Returns:
        np.ndarray: 描画後の画像
    """    
    red: Tuple[int, int, int] = (0, 0, 255)
    pt1_coordinate, pt2_coordinate = get_draw_info(pt1, pt2)
    cv2.line(image, pt1_coordinate, pt2_coordinate, red)
    return image

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
    connected.append((left_hand, left_elbow))
    connected.append((left_elbow, left_shoulder))
    return connected





