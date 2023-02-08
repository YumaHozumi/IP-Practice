import numpy as np

#画面の縮尺
SCALE_UP = 3.5

#重みベクトル
weight = np.ones(13)

#完全一致の場合のスコア
score_whole = np.sum(np.exp(weight * np.ones(13)))