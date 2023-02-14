from mytimer import Timer
import numpy as np

SCALE_UP = 3.5
TIMER = Timer()

weight = np.ones(13)

#完全一致の場合のスコア
score_perfect: np.ndarray = np.exp(weight * np.ones(13))