import cv2
from typing import Tuple, List
import numpy as np
from settings import SCALE_UP, X_LIMIT_START, Y_LIMIT_START


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
    cv2.line(image, pt1_coordinate, pt2_coordinate, red, thickness=3)
    return image

def create_connected(landmarks: List, index: int) -> List[Tuple[np.ndarray, np.ndarray]]:
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

    right_shoulder = landmarks[index].data[6]
    right_elbow = landmarks[index].data[8]
    right_hand = landmarks[index].data[10]
    
    left_hip = landmarks[index].data[11]
    left_knee = landmarks[index].data[13]
    left_ankle = landmarks[index].data[15]

    right_hip = landmarks[index].data[12]
    right_knee = landmarks[index].data[14]
    right_ankle = landmarks[index].data[16]
    connected.extend([(left_hand, left_elbow), (left_elbow, left_shoulder),
                        (right_hip, right_knee), (right_knee, right_ankle),
                        (left_hip, right_hip),(left_shoulder, left_hip),
                        (right_shoulder, right_hip), (right_hand, right_elbow), 
                        (right_elbow, right_shoulder), (left_shoulder, right_shoulder),
                        (left_hip, left_knee), (left_knee, left_ankle)])

    return connected

def created_three_connected(landmarks: np.ndarray, index: int
                        ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    connected: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    left_shoulder = landmarks[index].data[5]
    left_elbow = landmarks[index].data[7]
    left_hand = landmarks[index].data[9]

    right_shoulder = landmarks[index].data[6]
    right_elbow = landmarks[index].data[8]
    right_hand = landmarks[index].data[10]
    
    left_hip = landmarks[index].data[11]
    left_knee = landmarks[index].data[13]
    left_ankle = landmarks[index].data[15]

    right_hip = landmarks[index].data[12]
    right_knee = landmarks[index].data[14]
    right_ankle = landmarks[index].data[16]

    connected.extend([(left_elbow, left_hand, left_shoulder), 
                      (right_elbow, right_hand, right_shoulder),
                      (left_hip, left_knee, left_ankle),
                      (right_hip, right_knee, right_ankle)])
    return connected

def calculate_cos(pt1: np.ndarray, pt2: np.ndarray, pt3: np.ndarray) -> float:
    """cosを計算して角度を求める

    Args:
        pt1 (np.ndarray): 1つ目の点
        pt2 (np.ndarray): 2つ目の点
        pt3 (np.ndarray): 3つ目の点

    Returns:
        float: 角度
    """
    vec1: np.ndarray = pt2 - pt1
    vec2: np.ndarray = pt3 - pt1
    
    cos: float = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # print(cos)
    rad: float = np.arccos(cos)
    degree: float = np.rad2deg(rad)
    return degree

def draw_landmarks(image: np.ndarray, landmarks: List) -> np.ndarray:

    annotated_image = image.copy()
    # ランドマークとして検出されている点を囲む矩形を描画する
    # body_rectangle: List[float] = landmarks[0].json_data()["bbox"]
    # base_x, base_y, width, height = body_rectangle

    # x1 = int(base_x)
    # y1 = int(base_y - 10)
    # x2 = int(base_x+width)
    # y2 = int(base_x+height)
    # # 解像度1/4にしたので、4倍して位置を調整
    # cv2.rectangle(annotated_image, (x1*SCALE,y1*SCALE), (x2*SCALE, y2*SCALE), (255, 255, 255))

    connected_keypoints = create_connected(landmarks, index=0)
    for (pt1, pt2) in connected_keypoints:
        if((0 in pt1) or (0 in pt2)): continue # 座標をうまく取得できなかったとき
        annotated_image = draw_line(annotated_image, pt1, pt2)
    
    return annotated_image

def calc(landmarks: np.ndarray, index: int):
    connected: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = created_three_connected(landmarks, index)
    index = 0
    for (pt1, pt2, pt3) in connected:
        if((0 in pt1) or (0 in pt2) or (0 in pt3)): continue
        # print("index: ", end="")
        # print(index)
        # index += 1
        # print(calculate_cos(pt1, pt2, pt3))
