import numpy as np

"""
二人の部位ごとのベクトルの内積を取る
vec1:一人目のベクトル集合
vec2:二人目のベクトル集合
"""
def calculate_cos(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    cos: float = np.dot(vec1, vec2)
    return cos