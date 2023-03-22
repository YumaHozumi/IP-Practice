import cv2
import numpy as np
from typing import List, Tuple

#キャプチャーの用意
Capture = cv2.VideoCapture(0)

#入力画像の解像度を取得
Capture_Width = Capture.get(cv2.CAP_PROP_FRAME_WIDTH)
Capture_Height = Capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

#画面の縮尺
SCALE_UP = 3.5

#処理領域の設定(macでは画像サイズが1280×720)
X_LIMIT_START = 250
Y_LIMIT_START = 200
X_LIMIT_END = 1030
Y_LIMIT_END = 720

#結果の表示位置の設定
Result_X = 400
Result_Y = 150

# カウント文字の表示スピードに関してはここで調整
FRAMES_PER_SECOND = 30

"""
スコア計算用パラメータ
"""
#厳しさを調整するパラメータ
lenient = [1.0, 1.0, (1.0 - 1.0)] #甘め
default_strict = [0.95, 1.6, (1.0 - 0.95)]
strict = [0.95, 2.5, (1.0 - 0.95)]
strictest = [0.95, 4.0, (1.0 - 0.95)] #厳格
strict_weight: np.ndarray = np.array([strict, strict, strictest, strict, strictest, lenient, default_strict, 
                strict, default_strict, strict, lenient, lenient, lenient])
strict_weight = strict_weight.T
#スケール調整用パラメータ
scale_weight = strict_weight[0] / np.exp(strict_weight[1])
bias = strict_weight[2]


#影響度を調整するパラメータ
largest = 3.0 #影響度最大
larger = 1.5
default_impact = 1.0
small = 0.5 #影響度最小
score_weight: np.ndarray = np.array([small, larger, largest, larger, largest, default_impact, 
                            larger, larger, larger, larger, default_impact, small, small])
score_weight = score_weight

#完全一致の場合のスコア
score_perfect: np.ndarray = score_weight * (scale_weight * np.exp(strict_weight[1]) + bias)