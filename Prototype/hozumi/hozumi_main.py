import multiprocessing as mp
from multiprocessing import Queue;
import time
import cv2
from multi_process import countdown, capture_frames, take_screenshot, playerChange
from SynchronizeProcess import SharedRunning

def mouse_callback(event, x, y, flags, param):
    pass

if __name__ == '__main__':
     # キューの作成
    queue: Queue = Queue(maxsize=10)
    queue2: Queue = Queue(maxsize=10)
    running = SharedRunning()

    # ビデオキャプチャプロセスの作成
    video_process = mp.Process(target=capture_frames, args=(queue, running, queue2))
    video_process.start()
    # カウントダウンプロセスの初期化
    countdown_process = None

    while running.value:
        # キューからフレームを取得
        frame = queue.get()
        normal_frame = queue2.get()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESCキーが押されたら
            running.value = False

        if countdown_process is None:
            # カウントダウンプロセスが動作していない場合
            if key == 13:  # Enterキーが押されたら
                # カウントダウンプロセスの作成
                countdown_process = mp.Process(target=playerChange, args=(queue, running, queue2, 2))
                countdown_process.start()
                time.sleep(0.1)
        elif not countdown_process.is_alive():
            # カウントダウンプロセスが停止している場合
            countdown_process = None
            # screenshot_process = mp.Process(target=take_screenshot, args=(queue2, ))
            # screenshot_process.start()

    if countdown_process is not None:
        countdown_process.terminate()
    video_process.terminate()
    cv2.destroyAllWindows()
