import sys
import cv2
import numpy as np
import openpifpaf
from PIL import Image
from typing import List, Tuple
from vector_functions import correct_vectors
from draw_function import draw_vectors, draw_vectors_0, draw_result
from regist_functions import register
from display_functions import display_registered_playeres
from calculation import compare_pose, calc_multiSimilarity
from settings import SCALE_UP
from area_settings import Window_width, Window_height, human_width, humuan_height

def get_humanPicture(capture: cv2.VideoCapture, predictor: openpifpaf.predictor.Predictor, registeredNum: int, leader_Id: int) -> List[np.ndarray]:
    """leaderとplayerの写真と姿勢推定の結果を取得する

    Args:
        capture (cv2.VideoCapture): キャプチャー
        predictor (openpifpaf.predictor.Predictor): 姿勢推定モデル
        registeredNum (int): 登録されたプレイヤーの人数
        leader_Id (int): leaderのインデックス

    Returns:
        List[np.ndarray]: _description_
    """
    leader_finished: bool = False #leaderの姿勢登録が完了したかどうか

    #プレイヤーの登録が終了するまで登録作業を繰り返す
    while not leader_finished:
        #leaderのスクショを取得する
        leader_screen = capture_leader(capture)
        #対象領域のみ切り取る
        leader_picture = extract_leaderArea(leader_screen)

        #leaderの姿勢を推定する
        leader_pose, gt_anns, meta = predictor.numpy_image(leader_picture)
        if len(leader_pose) == 0:
            continue
        leader_vectors = correct_vectors(leader_pose, 0)
        leader_picture = draw_vectors_0(leader_picture, leader_vectors)

        #leaderのスクショを表示
        cv2.imshow('Camera 1',leader_picture)
        while True:
            key = cv2.waitKey(10)
            # Enterキーを押すと、
            if key == 0x0d:
                print('Save frame...')
                leader_finished = True
                break
            if key == 127:
                break
        
    #playerの姿勢を推定する
    playerNum = registeredNum - 1
    players_complete = False
    while not players_complete:
        players_complete = True
        player_screen = capture_players(capture, playerNum)
        player_pictures = extract_playersArea(player_screen, playerNum)
        players_vectors = []
        for i in range(playerNum):
            player_pose, gt_anns, meta = predictor.numpy_image(player_pictures[i])
            if len(player_pose) > 0:
                players_vectors.append(correct_vectors(player_pose, 0))
            else:
                players_complete = False #1人も検出されない画像があった場合、再度撮影を行う
                break
            player_pictures[i] = draw_vectors_0(player_pictures[i], players_vectors[i])

    if players_vectors:
        players_result = player_pictures[0]
        for j in range(len(players_vectors) - 1):
            players_result = cv2.hconcat([players_result, player_pictures[j + 1]]) 

        #playerのスクショを表示
        cv2.imshow('Camera 1',players_result)
        while True:
            # Enterキーを押すまで、スクショを表示
            if cv2.waitKey(10) == 0x0d:
                print('Save frame...')
                break
    
    return leader_picture, leader_vectors, player_pictures, players_vectors

def capture_leader(capture: cv2.VideoCapture) -> np.ndarray:
    """leaderのスクショを取る

    Args:
        capture (cv2.VideoCapture): キャプチャー

    Returns:
        np.ndarray: leaderのスクショ
    """
    #認識領域の設定
    area_Xstart = int((Window_width - human_width)/2)
    area_Xend = int((Window_width + human_width)/2)
    area_Ystart = Window_height - humuan_height
    area_Yend = Window_height

    print("Width:" + str(area_Xend - area_Xstart))
    print("Height:" + str(area_Yend - area_Ystart))

    while capture.isOpened():
        #success: 画像の取得が成功したか
        #frame: RGBの値を持っている3次元の配列データ ex) サイズ (480, 640, 3) 高さ、幅、色チャネル
        read_video: Tuple[bool, np.ndarray] = capture.read()
        success, frame = read_video
        # print("frame1 =",frame)

        if not success :
            print( "frame is None" )
            break
        
        annotated_image = frame.copy()
        #認識領域を表示
        annotated_image = cv2.resize(annotated_image, (Window_width, Window_height))
        annotated_image = cv2.rectangle(annotated_image, (area_Xstart, area_Ystart), (area_Xend, area_Yend), (0,255,0), thickness=2)
        annotated_image = cv2.flip(annotated_image, 1)

        cv2.imshow('Camera 1',annotated_image)
        #cv2.moveWindow("Camera 1", 200,40)

        # Enterキーを押したら画像の読み込みを終了
        if cv2.waitKey(10) == 0x0d:
            print('Enter pressed. Saving ...')
            break

    #Enter押下時のスクショを返す
    return cv2.resize(frame, (Window_width, Window_height))

def extract_leaderArea(frame: np.ndarray) -> np.ndarray:
    """leaderのスクショから、対象領域の画像を抽出

    Args:
        frame (np.ndarray): leaderのスクショ

    Returns:
        np.ndarray: 対象領域の画像
    """
    #認識領域の設定
    area_Xstart = int((Window_width - human_width)/2)
    area_Xend = int((Window_width + human_width)/2)
    area_Ystart = Window_height - humuan_height
    area_Yend = Window_height

    #領域の切り取り
    register_frame = frame[area_Ystart:area_Yend, area_Xstart:area_Xend]
    register_frame = cv2.flip(register_frame, 1)

    return register_frame


def capture_players(capture: cv2.VideoCapture, playerNum: int) -> List[np.ndarray]:
    """playerのスクショを取る

    Args:
        capture (cv2.VideoCapture): キャプチャー
        playerNum (int): プレイヤーの人数

    Returns:
        List[np.ndarray]: playerのスクショ
    """
    if playerNum > 3: playerNum = 3
    elif playerNum < 1: playerNum = 1

    #認識領域の設定
    area_Xstarts = []
    area_Xends = []

    for i in range(playerNum):
        area_Xstarts.append(int(((2*i+1)/playerNum * Window_width - human_width)/2))
        area_Xends.append(int(((2*i+1)/playerNum * Window_width + human_width)/2))

    area_Ystart = Window_height - humuan_height
    area_Yend = Window_height

    print("Width:" + str(area_Xends[0] - area_Xstarts[0]))
    print("Height:" + str(area_Yend - area_Ystart))

    while capture.isOpened():
        #success: 画像の取得が成功したか
        #frame: RGBの値を持っている3次元の配列データ ex) サイズ (480, 640, 3) 高さ、幅、色チャネル
        read_video: Tuple[bool, np.ndarray] = capture.read()
        success, frame = read_video
        # print("frame1 =",frame)

        if not success :
            print( "frame is None" )
            break
        
        annotated_image = frame.copy()
        annotated_image = cv2.resize(annotated_image, (Window_width, Window_height))
        #登録をおこなう領域を指定

        for j in range(playerNum):
            #認識領域を表示
            annotated_image = cv2.rectangle(annotated_image, (area_Xstarts[j], area_Ystart), (area_Xends[j], area_Yend), (0,255,0), thickness=2)
            
        annotated_image = cv2.flip(annotated_image, 1)

        cv2.imshow('Camera 1',annotated_image)
        #cv2.moveWindow("Camera 1", 200,40)

        # Enterキーを押したら画像の読み込みを終了
        if cv2.waitKey(10) == 0x0d:
            print('Enter pressed. Saving ...')
            break

    #Enter押下時のスクショを返す
    #return cv2.flip(frame, 1)
    return cv2.resize(frame, (Window_width, Window_height))

def extract_playersArea(frame: np.ndarray, playerNum: int) -> List[np.ndarray]:
    """playersのスクショから、対象領域の画像を抽出

    Args:
        frame (np.ndarray): playersのスクショ

    Returns:
        List[np.ndarray]: 対象領域の画像
    """
    if playerNum > 3: playerNum = 3
    elif playerNum < 1: playerNum = 1

    #認識領域の設定
    area_Xstarts = []
    area_Xends = []

    for i in range(playerNum):
        area_Xstarts.append(int(((2*i+1)/playerNum * Window_width - human_width)/2))
        area_Xends.append(int(((2*i+1)/playerNum * Window_width + human_width)/2))

    area_Ystart = Window_height - humuan_height
    area_Yend = Window_height

    #領域の切り取り
    #領域の切り取り
    register_frames:List = []
    for j in range(playerNum):
        register_frame = frame[area_Ystart:area_Yend, area_Xstarts[j]:area_Xends[j]]
        register_frames.append(cv2.flip(register_frame, 1))
        
    return register_frames