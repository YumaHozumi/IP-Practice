import multiprocessing as mp
from multiprocessing import Queue;
import time
import cv2
from multi_process import countdown, capture_frames

if __name__ == '__main__':
     # キューの作成
    queue: Queue = Queue(maxsize=10)
    running = mp.Value('i', True)

    # ビデオキャプチャプロセスの作成
    video_process = mp.Process(target=capture_frames, args=(queue, running))
    video_process.start()

    # カウントダウンプロセスの作成
    countdown_process = mp.Process(target=countdown, args=(queue, running))
    countdown_process.start()


    while running.value:
        # キューからフレームを取得
        frame = queue.get()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESCキーが押されたら

            running.value = False

    video_process.terminate()
    countdown_process.terminate()
    cv2.destroyAllWindows()
