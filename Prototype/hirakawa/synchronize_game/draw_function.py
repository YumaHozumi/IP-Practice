import cv2
import numpy as np
from typing import List, Tuple
from functions import get_draw_info, create_connected
from settings import SCALE_UP, Result_X, Result_Y


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


def draw_landmarks(image: np.ndarray, landmarks: List) -> np.ndarray:
    """骨格表示を追加する(create_connected使用版)

    Args:
        image (np.ndarray): 骨格を表示する画像
        landmarks (List): 認識結果のリスト

    Returns:
        np.ndarray: 骨格表示を追加した画像
    """

    annotated_image = image.copy()

    for people_num in range(len(landmarks)):
        connected_keypoints = create_connected(landmarks, index=people_num)
        for (pt1, pt2) in connected_keypoints:
            if((0 in pt1) or (0 in pt2)): continue # 座標をうまく取得できなかったとき

            annotated_image = draw_line(annotated_image, pt1, pt2)
    
    return annotated_image


def draw_vectors(image: np.ndarray, vectors: List) -> np.ndarray:
    """骨格表示を追加する(correct_vectors使用版)

    Args:
        image (np.ndarray): 骨格を表示する画像
        vectors (List): 関節点を結んだベクトルのリスト

    Returns:
        np.ndarray: 骨格表示を追加した画像
    """

    annotated_image = image.copy()

    for (pt1, pt2) in vectors:
            if((0 in pt1) or (0 in pt2)): continue # 座標をうまく取得できなかったとき

            annotated_image = draw_line(annotated_image, pt1, pt2)    
    
    return annotated_image


def draw_rectangle(image: np.ndarray, landmarks: List) -> np.ndarray:
    """認識した人を囲う矩形表示を追加する

    Args:
        image (np.ndarray): 矩形を表示する画像
        landmarks (List): 認識結果のリスト

    Returns:
        np.ndarray: 矩形表示を追加した画像
    """
    
    annotated_image = image.copy()
    
    #landmarks[0] 検出した一人目の
    #landmarks[0].data[0] keypoint番号0(鼻)の
    #landmarks[0].data[0][0] x座標
    #landmarks[0].data[0][1] y座標
    #landmarks[0].data[0][2] 正確さ的なやつ
    
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
    """認識した人のid表示を追加する

    Args:
        image (np.ndarray): id表示を行う画像
        landmarks (List): 認識結果のリスト
        image_width (int): 画像の幅

    Returns:
        np.ndarray: _description_
    """
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

def draw_similarity(image: np.ndarray, landmarks: List, index_1: int, index_2, similarity: float) -> np.ndarray:
    """指定した2人の頭上中央あたりに姿勢の類似度を表示する

    Args:
        image (np.ndarray): 類似度の表示を追加する画像
        landmarks (List): 認識結果のリスト
        index_1 (int): 1人目のid
        index_2 (_type_): 2人目のid
        similarity (float): 類似度

    Returns:
        np.ndarray: 類似度の表示を追加した画像
    """
    annotated_image = image.copy()
    body1_rectangle: List[float] = landmarks[index_1].json_data()["bbox"]
    base1_x, base1_y, area_width1, area_height1 = body1_rectangle
    body2_rectangle: List[float] = landmarks[index_2].json_data()["bbox"]
    base2_x, base2_y, area_width1, area_height1 = body2_rectangle

    x1: int = int(((base1_x + base2_x) / 2) * SCALE_UP)
    if(base1_y > base2_y):
        y1 = int((base2_y - 10) * SCALE_UP)
    else:
        y1 = int((base1_y - 10) * SCALE_UP)

    # 文字を重畳
    id_color = (255,255,255)
    id_txt = "similarity: " + str(similarity)
    cv2.putText(annotated_image,id_txt,(x1, y1),cv2.FONT_HERSHEY_SIMPLEX,1.0,id_color,2,cv2.LINE_4)
    

    return annotated_image


def draw_result(image: np.ndarray, similarity: float) -> np.ndarray:
    """
    結果の表示を行うメソッド

    Args:
        image: 結果の描画を追加する画像(フレーム)
        similarity: 表示する結果(類似度)

    Returns:
        np.ndarray: 結果表示を追加した画像
    """

    annotated_image = image.copy()

    # 文字を重畳
    id_color = (0,255,0)
    id_txt = "similarity: " + "{:.2f}".format(similarity)
    cv2.putText(annotated_image,id_txt,(Result_X, Result_Y),cv2.FONT_HERSHEY_SIMPLEX,2.0,id_color,2,cv2.LINE_4)

    return annotated_image

def draw_peopleNumber(image: np.ndarray, peopleNumber: int) -> np.ndarray:
    """人数を表示する

    Args:
        image (np.ndarray): 人数の描画を追加する画像
        peopleNumber (float): 人数

    Returns:
        np.ndarray: 人数表示を追加した画像
    """

    annotated_image = image.copy()

    # 文字を重畳
    id_color = (0,255,0)
    id_txt = "peopleNumber: " + str(peopleNumber)
    cv2.putText(annotated_image,id_txt,(Result_X, Result_Y),cv2.FONT_HERSHEY_SIMPLEX,2.0,id_color,2,cv2.LINE_4)

    return annotated_image

