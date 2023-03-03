import sys
import cv2
import numpy as np
import openpifpaf
from PIL import Image
from typing import List, Tuple
from functions import create_connected
from vector_functions import correct_vectors
from draw_function import draw_line, draw_landmarks,draw_vectors, draw_rectangle, draw_id, draw_similarity, draw_result
from calculation import compare_pose, calc_multiSimilarity
from settings import SCALE_UP, X_LIMIT_START, Y_LIMIT_START, X_LIMIT_END, Y_LIMIT_END


# PCに繋がっているUSBカメラから撮る場合はこれ
capture = cv2.VideoCapture(0)

if not capture.isOpened(): # 正常に動画読み込めなかったとき
    print( "Error opening capture device")
    capture.release() # カメラデバイス閉じる
    cv2.destroyAllWindows() # 開いているすべてのウィンドウ閉じる
    exit()

if capture.isOpened(): # 正常に読みこめたとき
    print( "Device captured correctly",capture)

predictor = openpifpaf.Predictor(checkpoint = "shufflenetv2k16")


while capture.isOpened():
    """
    success：画像の取得が成功したか
    frame：RGBの値を持っている3次元の配列データ ex) サイズ (480, 640, 3) 高さ、幅、色チャネル
    """
    read_video: Tuple[bool, np.ndarray] = capture.read()
    success, frame = read_video
    # print("frame1 =",frame)

    if not success :
        print( "frame is None" )
        break
    
    
        #認識を行う領域を制限
    limit_frame = frame[Y_LIMIT_START:Y_LIMIT_END, X_LIMIT_START:X_LIMIT_END]
    resize_frame: np.ndarray = cv2.resize(limit_frame, dsize=None, fx=(1.0 / SCALE_UP), fy=(1.0 / SCALE_UP))
    predictions, gt_anns, meta = predictor.numpy_image(resize_frame)
    """
    predictions：関節座標
    インデックス：関節座標点
    """
    
    annotated_image = frame.copy()

    #認識領域に人が映ってないときにもカメラ映像を出すように
    if len(predictions) == 0: 
        height = frame.shape[0]
        width = frame.shape[1]
        #サイズ確認用(テスト)
        #print('({0}, {1})'.format(height, width))
        
        annotated_image = cv2.rectangle(annotated_image, (X_LIMIT_START, Y_LIMIT_START), (X_LIMIT_END, Y_LIMIT_END), (0,255,0), thickness=2)
        annotated_image = cv2.flip(annotated_image, 1)
        bigger_frame = cv2.resize(annotated_image, (int(width) * 2, int(height) * 2))
        cv2.imshow('Camera 1',bigger_frame)
        
        # ESCキーを押すと終了
        if cv2.waitKey(100) == 0x1b:
            print('ESC pressed. Exiting ...')
            break
        
        continue

    #骨格を表示
    #annotated_image: np.ndarray = draw_landmarks(frame, predictions)
    #predictions[0].data[0] : (x,y,c)

    #骨格(ベクトル)を表示
    
    people_vectors: np.ndarray = np.zeros((len(predictions), 13, 2, 3))

    
    for person_id in range(len(predictions)):
        vectors = correct_vectors(predictions, person_id)
        annotated_image = draw_vectors(annotated_image, vectors)
        people_vectors[person_id] = np.asarray(vectors)

    #print(people_vectors.shape)
    
    #一人で検証する用
    #print(compare_pose(people_vectors[0], people_vectors[0]))
    

    #外接矩形を表示
    #annotated_image: np.ndarray = draw_rectangle(frame, predictions)

    height = frame.shape[0]
    width = frame.shape[1]
    annotated_image = cv2.rectangle(annotated_image, (X_LIMIT_START, Y_LIMIT_START), (X_LIMIT_END, Y_LIMIT_END), (0,255,0), thickness=2)
    annotated_image = cv2.flip(annotated_image, 1)

    #idの描画
    #annotated_image = draw_id(annotated_image, predictions, width)
    
    #similarityの描画
    if(len(predictions) >= 2):
        similarity = calc_multiSimilarity(people_vectors) * 100
        #print(similarity)
        #similarityの描画
        annotated_image = draw_result(annotated_image, similarity)

    bigger_frame = cv2.resize(annotated_image, (int(width) * 2, int(height) * 2))
    cv2.imshow('Camera 1',bigger_frame)
    #cv2.moveWindow("Camera 1", 200,40)

    # ESCキーを押すと終了
    if cv2.waitKey(100) == 0x1b:
        print('ESC pressed. Exiting ...')
        break

capture.release()
cv2.destroyAllWindows()