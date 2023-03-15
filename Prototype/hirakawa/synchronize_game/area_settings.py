import numpy as np

#ディスプレイウィンドウのサイズ
Window_width = 1280
Window_height = 720

#登録領域の設定(macでは画像サイズが1280×720)
X_LIMIT_START = 0
Y_LIMIT_START = 0
X_LIMIT_END = 1280
Y_LIMIT_END = 720

#人数の表示位置の設定
peopleNum_X = int(Window_width / 2)
peopleNum_Y = int(Window_height * 0.15)

#顔画像の幅と高さ
face_width = 200
face_height = 200

#人の姿勢推定領域のサイズ
human_width = int(Window_width / 4)
humuan_height = int(Window_height * 0.65)