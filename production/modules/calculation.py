import numpy as np
from vector_functions import convert_simpleVectors, normalize_vectors
from settings import score_perfect, strict_weight, scale_weight, bias, score_weight

def compare_pose(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """2人の姿勢の類似度を求める

    Args:
        vec1 (np.ndarray): 1人目のベクトル集合
        vec2 (np.ndarray): 2人目のベクトル集合

    Returns:
        float: 2人の姿勢の類似度
    """

    normalized_vec1 = normalize_vectors(convert_simpleVectors(vec1))
    xy_vectors_1 = normalized_vec1[:, :2] #xy成分だけ取り出す
    use_label_1 = normalized_vec1[:, 2] #ラベルを取り出す

    normalized_vec2 = normalize_vectors(convert_simpleVectors(vec2))
    xy_vectors_2 = normalized_vec2[:, :2] #xy成分だけ取り出す
    use_label_2 = normalized_vec2[:, 2] #ラベルを取り出す


    use_label_mixed: np.ndarray = use_label_1 * use_label_2


    #動作確認用
    #print(xy_vectors_1)
    #print(use_label_1)
    #print(xy_vectors_2)
    #print(use_label_2)
    #print(use_label_mixed)
    

    return calculate_score(xy_vectors_1, xy_vectors_2, use_label_mixed)


def calculate_score(xy_vectors_1: np.ndarray, xy_vectors_2: np.ndarray, label: np.ndarray) -> float:
    """類似度計算を行う(comparepose経由で呼び出す前提)

    Args:
        xy_vectors_1 (np.ndarray): 1人目のベクトルデータ
        xy_vectors_2 (np.ndarray): 2人目のベクトルデータ
        label (np.ndarray): ベクトルの検出・未検出のラベル

    Returns:
        float: 類似度
    """

    #内積を計算
    cos: np.ndarray = calculate_cos(xy_vectors_1, xy_vectors_2) 

    #コサイン値に重みづけ
    use_cos = cos * strict_weight[1]
    #指数関数化
    exp_cos = np.exp(use_cos)
    vector_points = score_weight * (scale_weight * exp_cos + bias)

    #検出できたベクトルのみスコアをカウントする
    sum_points = np.sum(vector_points * label)


    #動作検証用
    #print(cos)
    #print(use_cos)
    #print(exp_cos)
    #print(sum_points)


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
    #print(sum_penalty_score)
    #print(f"label:{label}")
    #print(f"score_perfect：{score_perfect}")
    #print(f"分子：{sum_points}")
    #print(f"分母：{score_whole}")

    return sum_points / score_whole


def calculate_cos(xy_vectors_1: np.ndarray, xy_vectors_2: np.ndarray) -> np.ndarray:
    """内積計算を行う

    Args:
        xy_vectors_1 (np.ndarray): 1人目のベクトルデータ
        xy_vectors_2 (np.ndarray): 2人目のベクトルデータ

    Returns:
        np.ndarray: 内積の集合
    """
    
    #二人の部位ごとのベクトルの内積を取る
    cos_all: np.ndarray = np.dot(xy_vectors_1, xy_vectors_2.T)

    #対角成分抽出
    cos: np.ndarray = np.zeros(len(xy_vectors_1))
    for vector_num in range(len(xy_vectors_1)):
        cos[vector_num] = cos_all[vector_num][vector_num]
    return cos


def calc_penalty(label: np.ndarray) -> float:
    """未検出のベクトルの数に応じたペナルティを計算

    Args:
        label (np.ndarray): ベクトルの検出・未検出のラベル

    Returns:
        float: ペナルティ
    """
    not_detect_sum: int = np.sum(label == 0)

    #テスト用
    #print(f"not：{not_detect_sum}")

    # シグモイド関数を利用してみる
    #penalty: float = sigmoid(not_detect_sum)

    # 指数関数を利用してみる
    penalty: float = np.exp2(not_detect_sum)
    return penalty


def calc_multiSimilarity(people_vectors: np.ndarray) -> float:
    """グループ全体(人数問わず)の平均類似度を求める

    Args:
        people_vectors (np.ndarray): 認識した全ての人のベクトルデータ

    Returns:
        float: グループ類似度(全体の平均類似度)
    """

    sum_similarity = 0  #総類似度
    num_pares = 0       #組み合わせ数

    for i in range(len(people_vectors)):
        for j in range(len(people_vectors)):
            if(i < j):
                sum_similarity += compare_pose(people_vectors[i], people_vectors[j])
                num_pares += 1
                #print(num_pares) #テスト用(組み合わせ数を表示)

    #平均類似度を返す
    return (sum_similarity / num_pares)