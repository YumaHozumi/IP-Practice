from mytimer import Timer
import numpy as np

SCALE_UP = 3.5
TIMER = Timer()
X_LIMIT_START = 170
Y_LIMIT_START = 100
X_LIMIT_END = 480
Y_LIMIT_END = 480

weight = np.ones(13)

#完全一致の場合のスコア
score_perfect: np.ndarray = np.exp(weight * np.ones(13))