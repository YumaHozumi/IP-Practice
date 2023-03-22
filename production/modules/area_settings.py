import numpy as np
from .settings import Capture_Width, Capture_Height

#ディスプレイウィンドウのサイズ
Window_width = 1920
Window_height = 1080

#登録領域の設定(macでは画像サイズが1280×720)
X_LIMIT_START = int(Capture_Width * 0.15)
Y_LIMIT_START = int(Capture_Height * 0.15)
X_LIMIT_END = int(Capture_Width * 0.85)
Y_LIMIT_END = int(Capture_Height)

#人数の表示位置の設定
peopleNum_X = int(Window_width / 2)
peopleNum_Y = int(Window_height * 0.15)

#顔画像の幅と高さ
face_width = 200
face_height = 200

#人の姿勢推定領域のサイズ
human_width = int(Capture_Width / 3.2)
human_height = int(Capture_Height * 0.7)