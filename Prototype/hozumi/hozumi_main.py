import sys
import cv2
import numpy as np
import openpifpaf
from PIL import Image
from typing import List, Tuple
from functions import draw_line, create_connected, calculate_cos, created_three_connected
from settings import SCALE_UP, TIMER, X_LIMIT_START, Y_LIMIT_START, X_LIMIT_END, Y_LIMIT_END, COUNT_X, COUNT_Y
from calculation import compare_pose
from vector_functions import correct_vectors
from threading import Timer
import threading
import time
import queue

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
q = queue.Queue()
frame_q = queue.Queue()
temp = None

def countDown(counts: int):
    global temp
    for i in range(counts+1): 
        time.sleep(1)
        print(i)
        q.put(counts-i)
    time.sleep(1)
    temp = None

def screenshot(frame: np.ndarray):
    cv2.imwrite(filename="test.png", img=frame)

while capture.isOpened():
    """
    success：画像の取得が成功したか
    frame：RGBの値を持っている3次元の配列データ ex) サイズ (480, 640, 3) 高さ、幅、色チャネル
    """

    read_video: Tuple[bool, np.ndarray] = capture.read()
    success, frame = read_video

    if not success :
        print( "frame is None" )
        break
    # img[top : bottom, left : right]
    limit_frame = frame[Y_LIMIT_START:Y_LIMIT_END, X_LIMIT_START:X_LIMIT_END]
    #limit_frame = frame
    resize_frame: np.ndarray = cv2.resize(limit_frame, dsize=None, fx=(1.0 / SCALE_UP), fy=(1.0 / SCALE_UP))

    predictions, gt_anns, meta = predictor.numpy_image(resize_frame)
    """
    predictions：関節座標
    インデックス：関節座標点
    """
    if len(predictions) == 0: continue
    annotated_image: np.ndarray = draw_landmarks(frame, predictions)
    #predictions[0].data[0] : (x,y,c)

    people_vectors: np.ndarray = np.zeros((len(predictions), 13, 2, 3))

    for person_id in range(len(predictions)):
        vectors = correct_vectors(predictions, person_id)
        people_vectors[person_id] = np.asarray(vectors)

    if len(people_vectors) >= 1:
        similarity = compare_pose(people_vectors[0], people_vectors[0]) * 100
        # print(f"類似度：{similarity}")
        # print("---------------------------")

    height = frame.shape[0]
    width = frame.shape[1]
    annotated_image = cv2.rectangle(annotated_image, (X_LIMIT_START, Y_LIMIT_START), (X_LIMIT_END, Y_LIMIT_END), (0,255,0), thickness=2)
    annotated_image = cv2.flip(annotated_image, 1)

    # print("frame1 =",frame)

    if not q.empty():
        temp = q.get()
        if temp == 0:
            pic_thread = threading.Thread(target=screenshot, args=(frame, ))
            pic_thread.start()
            pic_thread.join()

        print(f"time: {temp}")
    if temp != None:
        cv2.putText(annotated_image, text=f"count: {temp}", org=(COUNT_X, COUNT_Y), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=2.0, color=(0,255,0), thickness=2,lineType=cv2.LINE_4)

    bigger_frame = cv2.resize(annotated_image, (int(width) * 2, int(height) * 2))
    cv2.imshow('Camera 1',bigger_frame)
    #cv2.moveWindow("Camera 1", 200,40)
    calc(predictions, 0)

    # TIMER.end()
    # TIMER.calc_speed()

    # ESCキーを押すと終了
    
    # 数値100で0.1s秒キー入力待つ
    # ※ここがボトルネックの一因になってた可能性あり※
    # 100->1でめっちゃぬるぬる動くように、、、
    if cv2.waitKey(1) == 0x1b:
        print('ESC pressed. Exiting ...')
        break

    # タイマーの計測開始
    # TIMER.start()
    if cv2.waitKey(1) == ord('c'):
        if threading.active_count() <= 1:
            thread = threading.Thread(target=countDown, args=(5,))
            thread.setDaemon(True)
            thread.start()
capture.release()
cv2.destroyAllWindows()

