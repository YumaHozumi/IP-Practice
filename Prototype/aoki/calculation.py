import numpy as np
from vector_functions import convert_simpleVectors, normalize_vectors
from settings import weight, score_whole

def normalize_vector():
    return 0

def compare_pose(vec1: np.ndarray, vec2: np.ndarray):
    normalized_vec1 = normalize_vectors(convert_simpleVectors(vec1))
    xy_vectors_1 = normalized_vec1[:, :2] #xy成分だけ取り出す
    use_label_1 = normalized_vec1[:, 2] #ラベルを取り出す

    normalized_vec2 = normalize_vectors(convert_simpleVectors(vec2))
    xy_vectors_2 = normalized_vec2[:, :2] #xy成分だけ取り出す
    use_label_2 = normalized_vec2[:, 2] #ラベルを取り出す

    #print(xy_vectors_1)
    print(use_label_1)
    #print(xy_vectors_2)
    print(use_label_2)

    use_label_mixed: np.ndarray = use_label_1 * use_label_2

    print(use_label_mixed)

    return calculate_score(xy_vectors_1, xy_vectors_2, use_label_mixed)


def calculate_score(xy_vectors_1: np.ndarray, xy_vectors_2: np.ndarray, label: np.ndarray):
    #内積を計算
    cos: np.ndarray = calculate_cos(xy_vectors_1, xy_vectors_2) 
    print(cos)

    #コサイン値に重みづけ
    use_cos = cos * weight
    #指数関数化
    exp_cos = np.exp(use_cos)
    print(use_cos)
    print(exp_cos)

    #検出できたベクトルのみスコアを加算していく
    sum_points = 0
    for point_num in range(len(exp_cos)):
        if(not (label[point_num] == 0)):
            sum_points += exp_cos[point_num]


    return sum_points / score_whole

"""
二人の部位ごとのベクトルの内積を取る
vec1:一人目のベクトル集合
vec2:二人目のベクトル集合
"""
def calculate_cos(xy_vectors_1: np.ndarray, xy_vectors_2: np.ndarray) -> np.ndarray:
    cos_all: np.ndarray = np.dot(xy_vectors_1, xy_vectors_2.T)
    cos: np.ndarray = np.zeros(len(xy_vectors_1))

    #対角成分抽出
    for vector_num in range(len(xy_vectors_1)):
        cos[vector_num] = cos_all[vector_num][vector_num]
    return cos