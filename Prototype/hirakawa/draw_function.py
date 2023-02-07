import cv2
import numpy as np
from typing import List, Tuple
from functions import get_draw_info, draw_line, create_connected

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


def draw_rectangle(image: np.ndarray, landmarks: List) -> np.ndarray:
    annotated_image = image.copy()
    """
    landmarks[0] 検出した一人目の
    landmarks[0].data[0] keypoint番号0(鼻)の
    landmarks[0].data[0][0] x座標
    landmarks[0].data[0][1] y座標
    landmarks[0].data[0][2] 正確さ的なやつ
    """
    # ランドマークとして検出されている点を囲む矩形を描画する
    body_rectangle: List[float] = landmarks[0].json_data()["bbox"]
    base_x, base_y, width, height = body_rectangle

    x1 = int(base_x)
    y1 = int(base_y - 10)
    x2 = int(base_x+width)
    y2 = int(base_x+height)
    # 解像度1/4にしたので、4倍して位置を調整
    points = get_draw_info([x1,y1], [x2, y2])
    cv2.rectangle(annotated_image, points[0], points[1], (255, 255, 255))
    
    return annotated_image