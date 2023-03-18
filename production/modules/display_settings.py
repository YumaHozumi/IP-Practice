import numpy as np
from typing import List, Tuple
from area_settings import Window_height, Window_width

#プレイヤーを識別する色(B,G,R)
player_color: List = [(0, 0, 255),(255, 0, 0),(0, 255, 0),(0, 165, 255)]

#諸々表示する際の背景画像(白一色)
whiteboard = 255 * np.ones([Window_height, Window_width, 3])