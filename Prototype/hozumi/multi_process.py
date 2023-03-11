import cv2
import numpy as np
import openpifpaf
from typing import List, Tuple
from functions import draw_landmarks
from settings import SCALE_UP, TIMER, X_LIMIT_START, Y_LIMIT_START, X_LIMIT_END, Y_LIMIT_END, COUNT_X, COUNT_Y
from calculation import compare_pose
from vector_functions import correct_vectors
import multiprocessing as mp

def add_countdown(frame: np.ndarray, count: int):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if count >= 0:
        cv2.putText(frame, str(count), (300, 300), font, 7, (0, 255, 255), 10, cv2.LINE_AA)
    else:
        cv2.putText(frame, '', (300, 300), font, 7, (0, 255, 255), 10, cv2.LINE_AA)
    return frame

def countdown(queue: mp.Queue, running, count: int):
    # 5秒カウントダウン
    for i in range(count, -1, -1):
        print(i)
        for j in range(30):
            if not running.value:
                return         
            frame = queue.get()
            frame = add_countdown(frame, i)
            queue.put(frame)
    return 

def take_screenshot(q2: mp.Queue) -> None:
    frames = []
    while not q2.empty():
        frame = q2.get()
        frames.append(frame)
    if frames:
        #cv2.imwrite(filename="screenshot.png", img=frames[-1])
        pass

def capture_frames(queue: mp.Queue, running, q2: mp.Queue):

    predictor = openpifpaf.Predictor(checkpoint = "shufflenetv2k16")

    # PCに繋がっているUSBカメラから撮る場合はこれ
    capture = cv2.VideoCapture(0)

    if not capture.isOpened(): # 正常に動画読み込めなかったとき
        print( "Error opening capture device")
        capture.release() # カメラデバイス閉じる
        cv2.destroyAllWindows() # 開いているすべてのウィンドウ閉じる
        exit()

    if capture.isOpened(): # 正常に読みこめたとき
        print( "Device captured correctly",capture)


    while capture.isOpened():
        """
        success：画像の取得が成功したか
        frame：RGBの値を持っている3次元の配列データ ex) サイズ (480, 640, 3) 高さ、幅、色チャネル
        """
        if not running.value: return
        read_video: Tuple[bool, np.ndarray] = capture.read()
        success, frame = read_video
        if not success :
            print( "frame is None" )
            break
        q2.put(frame)
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

        bigger_frame = cv2.resize(annotated_image, (int(width) * 2, int(height) * 2))
        queue.put(bigger_frame)


    # タイマーの計測開始
    # TIMER.start()
    capture.release()

