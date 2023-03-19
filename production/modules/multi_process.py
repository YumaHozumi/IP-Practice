import cv2
import numpy as np
import openpifpaf
from typing import List, Tuple
#from functions import draw_landmarks
from .settings import SCALE_UP, X_LIMIT_START, Y_LIMIT_START, X_LIMIT_END, Y_LIMIT_END, FRAMES_PER_SECOND
from .calculation import compare_pose
from .vector_functions import correct_vectors
import multiprocessing as mp
import time

def add_countdown(frame: np.ndarray, count: int) -> np.ndarray:
    """カウントをフレーム上に追加するメソッド

    Args:
        frame (np.ndarray): カウントを載せたいフレーム
        count (int): 今何秒目か

    Returns:
        np.ndarray: カウントの情報をフレームに追加したもの
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    if count >= 0:
        cv2.putText(frame, str(count), (300, 300), font, 7, (0, 255, 255), 10, cv2.LINE_AA)
    else:
        cv2.putText(frame, '', (300, 300), font, 7, (0, 255, 255), 10, cv2.LINE_AA)
    return frame

def countdown(queue: mp.Queue, running, count: int):
    """カウントダウンを行うためのメソッド

    Args:
        queue (mp.Queue): フレーム（加工済み）を格納するキュー
        running (boolean): プログラムが実行中か
        count (int): 何秒間カウントするか
    """
    frames_per_count = count * FRAMES_PER_SECOND
    start_time = time.monotonic()
    for i in range(frames_per_count + 1):
        if not running.value:
            return
        frame_time = i / FRAMES_PER_SECOND
        current_time = time.monotonic() - start_time
        if frame_time > current_time:
            time.sleep(frame_time - current_time)
        frame = queue.get()
        frame = add_countdown(frame, count - (i // FRAMES_PER_SECOND))
        queue.put(frame)
    return

def playerChange(queue: mp.Queue, running, q2: mp.Queue, changeNum: int):
    """lead役->スクショ->Player役->スクショ の1セットをchangeNum回繰り返すメソッド

    Args:
        queue (mp.Queue): フレーム（加工済み）を格納するキュー
        running (boolean): プログラムが実行中か
        q2 (mp.Queue): フレーム（未加工）を格納するキュー
        changeNum (int): 何回セットを繰り返すか
    """
    for i in range(changeNum):
        print(f"{i+1}回目")
        for _ in range(2):
            process = mp.Process(target=countdown, args=(queue, running, 3))
            process.start()
            process.join()
            scrennshot_process = mp.Process(target=take_screenshot, args=(q2, ))
            scrennshot_process.start()
            scrennshot_process.join()

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

        annotated_image = frame.copy()
        """
        predictions：関節座標
        インデックス：関節座標点
        """

        #annotated_image: np.ndarray = draw_landmarks(frame, predictions)
        #predictions[0].data[0] : (x,y,c)

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

