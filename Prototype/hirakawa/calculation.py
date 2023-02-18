import numpy as np
from vector_functions import convert_simpleVectors, normalize_vectors
from settings import score_perfect, strict_weight, scale_weight, bias, score_weight

def compare_pose(vec1: np.ndarray, vec2: np.ndarray):
    normalized_vec1 = normalize_vectors(convert_simpleVectors(vec1))
    xy_vectors_1 = normalized_vec1[:, :2] #xy成分だけ取り出す
    use_label_1 = normalized_vec1[:, 2] #ラベルを取り出す

    normalized_vec2 = normalize_vectors(convert_simpleVectors(vec2))
    xy_vectors_2 = normalized_vec2[:, :2] #xy成分だけ取り出す
    use_label_2 = normalized_vec2[:, 2] #ラベルを取り出す

    

    use_label_mixed: np.ndarray = use_label_1 * use_label_2

    """
    #動作確認用
    print(xy_vectors_1)
    print(use_label_1)
    print(xy_vectors_2)
    print(use_label_2)
    print(use_label_mixed)
    """

    return calculate_score(xy_vectors_1, xy_vectors_2, use_label_mixed)


def calculate_score(xy_vectors_1: np.ndarray, xy_vectors_2: np.ndarray, label: np.ndarray):
    #内積を計算
    cos: np.ndarray = calculate_cos(xy_vectors_1, xy_vectors_2) 

    #コサイン値に重みづけ
    use_cos = cos * strict_weight[1]
    #指数関数化
    exp_cos = np.exp(use_cos)
    vector_points = score_weight * (scale_weight * exp_cos + bias)

    #検出できたベクトルのみスコアをカウントする
    sum_points = np.sum(vector_points * label)

    """
    動作検証用
    print(cos)
    print(use_cos)
    print(exp_cos)
    print(sum_points)
    """

    # ラベルの01反転
    not_detect_label: np.ndarray = np.logical_not(label)
    not_detect_score: np.ndarray = not_detect_label * score_perfect * calc_penalty(label)
    sum_penalty_score: float = not_detect_score.sum()
    
    #完全一致の場合のスコアを算出
    score_whole = np.sum(score_perfect * label)
    #未検出のベクトルにペナルティを入れる場合は下を使う
    #score_whole = np.sum(score_perfect)

    # ペナルティを加算
    score_whole += calc_penalty(label)

    #テスト用
    #print(not_detect_score)
    print(sum_penalty_score)
    # print(f"label:{label}")
    # print(f"score_perfect：{score_perfect}")
    print(f"分子：{sum_points}")
    print(f"分母：{score_whole}")

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

"""
ペナルティの計算
"""
def calc_penalty(label: np.ndarray) -> float:
    not_detect_sum: int = np.sum(label == 0)

    #テスト用
    print(f"not：{not_detect_sum}")

    # シグモイド関数を利用してみる
    #penalty: float = sigmoid(not_detect_sum)

    # 指数関数を利用してみる
    penalty: float = np.exp2(not_detect_sum)
    return penalty