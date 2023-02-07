import cv2
import numpy as np
from typing import List, Tuple
from functions import get_draw_info, draw_line, create_connected
from settings import SCALE_UP


def draw_landmarks(image: np.ndarray, landmarks: List) -> np.ndarray:

    annotated_image = image.copy()

    for people_num in range(len(landmarks)):
        connected_keypoints = create_connected(landmarks, index=people_num)
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
    for person_id in range(len(landmarks)):
        body_rectangle: List[float] = landmarks[person_id].json_data()["bbox"]
        base_x, base_y, width, height = body_rectangle

        x1 = int(base_x)
        y1 = int(base_y - 10)
        x2 = int(base_x+width)
        y2 = int(base_x+height)
        # 元の解像度での座標に戻して表示
        points = get_draw_info([x1,y1], [x2, y2])
        cv2.rectangle(annotated_image, points[0], points[1], (255, 255, 255))
    
    return annotated_image


def draw_id(image: np.ndarray, landmarks: List, image_width: int) -> np.ndarray:
    annotated_image = image.copy()

    for person_id in range(len(landmarks)):
        body_rectangle: List[float] = landmarks[person_id].json_data()["bbox"]
        base_x, base_y, area_width, area_height = body_rectangle

        x1 = int(image_width - ((base_x + area_width) * SCALE_UP))
        y1 = int((base_y - 20) * SCALE_UP)

        # 文字を重畳
        id_color = (255,255,255)
        id_txt = "id: " + str(person_id)
        cv2.putText(annotated_image,id_txt,(x1, y1),cv2.FONT_HERSHEY_SIMPLEX,1.0,id_color,2,cv2.LINE_4)
    
    return annotated_image

