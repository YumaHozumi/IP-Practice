import multiprocessing as mp
from multiprocessing import Queue;
import time
import cv2
from multi_process import countdown, capture_frames
from SynchronizeProcess import SharedRunning

if __name__ == '__main__':
     # キューの作成
    queue: Queue = Queue(maxsize=10)
    running = SharedRunning()

    # ビデオキャプチャプロセスの作成
    video_process = mp.Process(target=capture_frames, args=(queue, running))
    video_process.start()
    # カウントダウンプロセスの初期化
    countdown_process = None

    while running.value:
        # キューからフレームを取得
        frame = queue.get()
        cv2.imshow('frame', frame)
        if cv2.waitKeyEx(1) & 0xFF == 27:  # ESCキーが押されたら
            running.value = False

        if countdown_process is None:
            # カウントダウンプロセスが動作していない場合
            if cv2.waitKeyEx(1) == 13:  # Enterキーが押されたら
                # カウントダウンプロセスの作成
                countdown_process = mp.Process(target=countdown, args=(queue, running))
                countdown_process.start()
        elif not countdown_process.is_alive():
            # カウントダウンプロセスが停止している場合
            countdown_process = None


    video_process.terminate()
    if countdown_process is not None:
        countdown_process.terminate()
    cv2.destroyAllWindows()
