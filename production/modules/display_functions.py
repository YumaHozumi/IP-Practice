import cv2
import os
import numpy as np
import time
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
from .functions import get_draw_info, create_connected
from .settings import SCALE_UP, Result_X, Result_Y
from .display_settings import player_color
from .area_settings import X_LIMIT_START, Y_LIMIT_START, X_LIMIT_END, Y_LIMIT_END, face_width, face_height, Window_width, Window_height
from .area_settings import human_width, human_height, display_face_width, display_face_height, display_human_height, display_human_width
from pathlib import Path

whiteboard = 255 * np.ones([Window_height, Window_width, 3])

def display_title():
    """タイトルやデモの概要等を表示する
    """

    #print(os.getcwd()) #カレントディレクトリの確認 -> main.py実行時、productionから相対パスで指定すればOK
    #タイトル画像の用意
    title = cv2.imread('./modules/pictures/Title.bmp', cv2.IMREAD_COLOR )
    title = cv2.resize(title, (Window_width, Window_height))
    #ガイダンス画像の用意
    guidance = cv2.imread('./modules/pictures/guidance.bmp', cv2.IMREAD_COLOR )
    guidance = cv2.resize(guidance, (Window_width, Window_height))
    #ゲーム説明画像の用意
    game_abstract = cv2.imread('./modules/pictures/Game_abstract.bmp', cv2.IMREAD_COLOR )
    game_abstract = cv2.resize(game_abstract, (Window_width, Window_height))


    #タイトルを表示
    cv2.imshow('Camera 1', title)
    while True:
        if cv2.waitKey(10) == 0x0d: 
            break

    #ガイダンスを表示
    cv2.imshow('Camera 1', guidance)
    while True:
        if cv2.waitKey(10) == 0x0d: 
            break

    #ゲーム説明を表示
    cv2.imshow('Camera 1', game_abstract)
    while True:
        if cv2.waitKey(10) == 0x0d: 
            break

    return



def display_registered_playeres(face_Imgs: List[np.ndarray]) -> np.ndarray:
    """登録されたプレイヤー一覧を表示する

    Args:
        face_Imgs (List[np.array]): 顔画像のリスト

    Returns:
        np.ndarray: 一覧を表示している画像
    """

    #背景の設定
    playeresImg = cv2.imread('./modules/pictures/Registerd_players.bmp', cv2.IMREAD_COLOR )
    playeresImg = cv2.resize(playeresImg, (Window_width, Window_height)) 
    
    #登録結果表示画面の作成
    people_num = len(face_Imgs)
    for i in range(people_num):
        img = face_Imgs[i] #顔画像

        #描画領域の指定
        separate_width = Window_width / (people_num + 1)
        x_offset=int((i+1)*separate_width - display_face_width/2)
        y_offset=int(Window_height/2 - display_face_height/2)
        playeresImg[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img.copy()
        txt = "Player" + str(int(i + 1))
        #cv2.putText(playeresImg, txt, (x_offset, y_offset - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.75, player_color[i], 3, cv2.LINE_AA)
        cv2_putText(playeresImg, txt, (int(x_offset + display_face_width/2), int(y_offset - 60)), 100, player_color[i], 2)
        
    #型変換
    playeresImg = playeresImg.astype('uint8')

    cv2.imshow('Camera 1',playeresImg) #認識した顔の画像を表示

    return playeresImg

def display_instraction_leader(leader_id: int) -> np.ndarray:
    """leader役のプレイヤーへの指示を表示

    Args:
        leader_id (int): leader役のプレイヤーのid

    Returns:
        np.ndarray: 確認画面
    """
    
    instraction_Img = whiteboard.copy() #背景の設定

    #画面の説明の表示
    instraction_message: str = 'Player' + str(leader_id + 1) + 'さんがお手本役です。\n'
    instraction_message += '次の画面で表示される枠内で\nお手本になるポーズをとってください'
    cv2_putText(instraction_Img, instraction_message, (int(Window_width/2), int(Window_height/2)), 60, (0,0,0), 2)
    cv2_putText(instraction_Img, 'Start!  > Enter', (int(Window_width * 0.8), Window_height - 10), 40, (0,0,0))
    
    #型変換
    instraction_Img = instraction_Img.astype('uint8')
    #確認画面を表示
    cv2.imshow('Camera 1',instraction_Img) 

    return instraction_Img

def display_leaderRecognitionError() -> np.ndarray:
    """leaderを認識できなかったため再度playerの撮影を行うメッセージの表示を行う

    Returns:
        np.ndarray: メッセージ表示画面
    """
    
    error_Img = whiteboard.copy() #背景の設定

    #画面の説明の表示
    errorMessage: str = '認識エラーが発生しました。\nもう一度行います。'
    cv2_putText(error_Img, errorMessage, (int(Window_width/2), int(Window_height/2)), 80, (0,0,0), 2)
    cv2_putText(error_Img, '次へ  > Enter', (int(Window_width * 0.8), Window_height - 10), 40, (0,0,0))

    #型変換
    error_Img = error_Img.astype('uint8')
    #確認画面を表示
    cv2.imshow('Camera 1',error_Img) 

    return error_Img

def display_change() -> np.ndarray:
    """leaderを認識できなかったため再度playerの撮影を行うメッセージの表示を行う

    Returns:
        np.ndarray: メッセージ表示画面
    """
    
    error_Img = whiteboard.copy() #背景の設定

    #画面の説明の表示
    errorMessage: str = '見本役を交代します'
    cv2_putText(error_Img, errorMessage, (int(Window_width/2), int(Window_height/2)), 80, (0,0,0), 2)
    cv2_putText(error_Img, '次へ  > Enter', (int(Window_width * 0.8), Window_height - 10), 40, (0,0,0))

    #型変換
    error_Img = error_Img.astype('uint8')
    #確認画面を表示
    cv2.imshow('Camera 1',error_Img) 

    return error_Img

def display_check_leader(leader_picture: np.ndarray, leader_id: int) -> np.ndarray:
    """leader役のプレイヤーにポーズの確認をとる画面を表示

    Args:
        leader_picture (np.array): leader役のキャプチャーされた画像
        leader_id (int): leader役のプレイヤーのid

    Returns:
        np.ndarray: 確認画面
    """
    
    check_Img = whiteboard.copy() #背景の設定

    #画面の説明の表示
    page_about: str = 'プレイヤー' + str(leader_id + 1) + 'さんのお手本ポーズ'
    cv2_putText(check_Img, page_about, (20, 80), 80, (0,0,0))
    cv2_putText(check_Img, '　OK!　  > Enter', (int(Window_width * 0.7), Window_height - 50), 40, (0,0,0))
    cv2_putText(check_Img, 'やり直す > Delete', (int(Window_width * 0.7), Window_height - 10), 40, (0,0,0))
    
    #ポーズ確認画面の作成
    
    #表示するポーズ画像のサイズ調整
    leader_picture = cv2.resize(leader_picture, (display_human_width, display_human_height))
    
    #描画領域の指定
    picture_height = leader_picture.shape[0]
    picture_width = leader_picture.shape[1]
    x_offset=int(Window_width/2 - picture_width/2)
    y_offset=int(Window_height/2 - picture_height/2)
    check_Img[y_offset:y_offset+picture_height, x_offset:x_offset+picture_width] = leader_picture.copy()
    txt = "Player" + str(leader_id + 1)
    #プレイヤー番号の表示
    #cv2_putText(check_Img, txt, (int(x_offset + picture_width/2), y_offset - 40), 60, player_color[leader_id], 2)

    #型変換
    check_Img = check_Img.astype('uint8')
    #確認画面を表示
    cv2.imshow('Camera 1',check_Img) 

    return check_Img

def display_instraction_players(leader_picture: np.ndarray, leader_id: int) -> np.ndarray:
    """真似するポーズを表示し、player役のプレイヤーへの指示を表示

    Args:
        leader_picture (np.array): leader役のキャプチャーされた画像
        leader_id (int): leader役のプレイヤーのid

    Returns:
        np.ndarray: 確認画面
    """
    
    instraction_Img = whiteboard.copy() #背景の設定

    #画面の説明の表示
    page_about: str = 'このポーズをよく覚えてください。'
    cv2_putText(instraction_Img, page_about, (20, 80), 80, (0,0,0))
    cv2_putText(instraction_Img, 'Start!  > Enter', (int(Window_width * 0.8), Window_height - 10), 40, (0,0,0))
    
    #指示画面の作成

    #表示するポーズ画像のサイズ調整
    leader_picture = cv2.resize(leader_picture, (display_human_width, display_human_height))
    
    #描画領域の指定
    picture_height = leader_picture.shape[0]
    picture_width = leader_picture.shape[1]
    x_offset=int(Window_width/2 - picture_width/2)
    y_offset=int(Window_height/2 - picture_height/2)
    instraction_Img[y_offset:y_offset+picture_height, x_offset:x_offset+picture_width] = leader_picture.copy()

    #型変換
    instraction_Img = instraction_Img.astype('uint8')
    #確認画面を表示
    cv2.imshow('Camera 1',instraction_Img) 

    return instraction_Img

def display_result(player_pictures: np.ndarray, leader_id: int, players_id: List[int], similarities: List) -> np.ndarray:
    """1ゲームの結果を表示する

    Args:
        player_pictures (np.array): player役のキャプチャーされた画像
        leader_id (int): leader役のプレイヤーのid
        players_id (List[int]): player役のプレイヤーのid
        similarities (List): 類似度のリスト

    Returns:
        np.ndarray: 結果表示画面
    """
    
    result_Img = whiteboard.copy() #背景の設定

    #画面の説明の表示
    message: str = 'ゲーム' + str(leader_id + 1) + "の結果発表!"
    cv2_putText(result_Img, message, (20, 80), 80, (0,0,0))
    cv2_putText(result_Img, '次へ  > Enter', (int(Window_width * 0.8), Window_height - 10), 40, (0,0,0))
    
    #結果表示画面の作成
    player_num = len(player_pictures)
    #縮尺
    small_width = display_human_width
    small_hight = display_human_height

    area_Xstarts = []
    area_Xends = []

    for i in range(player_num):
        area_Xstarts.append(int(((2*i+1)/player_num * Window_width - small_width)/2))
        area_Xends.append(int(((2*i+1)/player_num * Window_width + small_width)/2))

    area_Ystart = int(Window_height*0.45 - small_hight/2)
    area_Yend = area_Ystart + small_hight

    for j in range(player_num):
        img = cv2.resize(player_pictures[j], (small_width, small_hight))
        result_Img[area_Ystart:area_Yend, area_Xstarts[j]:area_Xends[j]] = img.copy()
        result_Img = cv2.rectangle(result_Img, (area_Xstarts[j], area_Ystart), (area_Xends[j], area_Yend), player_color[players_id[j]], thickness=4)
        txt_X = int((area_Xstarts[j] + area_Xends[j])/2)
        txt_Y = int(area_Yend + 80)
        cv2_putText(result_Img, '{:.2f}'.format(similarities[j]), (txt_X, txt_Y), 80, (0,0,0), 2)

    #型変換
    result_Img = result_Img.astype('uint8')
    #確認画面を表示
    cv2.imshow('Camera 1',result_Img) 

    return result_Img

def display_playersRecognitionError() -> np.ndarray:
    """playerの内、認識できなかったため再度playerの撮影を行うメッセージの表示を行う

    Returns:
        np.ndarray: メッセージ表示画面
    """
    
    error_Img = whiteboard.copy() #背景の設定

    #画面の説明の表示
    errorMessage: str = '認識エラーが発生しました。\nもう一度行います。'
    cv2_putText(error_Img, errorMessage, (int(Window_width/2), int(Window_height/2)), 80, (0,0,0), 2)
    cv2_putText(error_Img, '次へ  > Enter', (int(Window_width * 0.8), Window_height - 10), 40, (0,0,0))

    #型変換
    error_Img = error_Img.astype('uint8')
    #確認画面を表示
    cv2.imshow('Camera 1',error_Img) 

    return error_Img

def display_final_result(face_Imgs: List[np.ndarray], similarities: List) -> np.ndarray:
    """最終結果を表示する

    Args:
        face_Imgs (List[np.array]): 顔画像のリスト
        similarities (List): 全てのゲーム結果

    Returns:
        np.ndarray: 一覧を表示している画像
    """
    resultImg = whiteboard.copy() #背景の設定

    #画面の説明の表示
    cv2_putText(resultImg, '最終結果', (20, 80), 80, (0,0,0))
    cv2_putText(resultImg, '　終了!　  > Enter', (int(Window_width * 0.7), Window_height - 10), 40, (0,0,0))
    
    #登録結果表示画面の作成
    people_num = len(face_Imgs)
    final_similarities = []

    for i in range(people_num):
        img = face_Imgs[i] #顔画像

        #描画領域の指定
        separate_width = Window_width / (people_num + 1)
        x_offset=int((i+1)*separate_width - display_face_width/2)
        y_offset=int(Window_height/2 - display_face_height/2)
        resultImg[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img.copy()
        txt = "Player" + str(int(i + 1))
        #cv2.putText(resultImg, txt, (x_offset, y_offset - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.75, player_color[i], 3, cv2.LINE_AA)
        cv2_putText(resultImg, txt, (int(x_offset + display_face_width/2), int(y_offset - 60)), 100, player_color[i], 2)
        if (people_num -1) == 1: 
            average_sim = similarities[i][0]
        elif (people_num -1) > 1:
            average_sim = sum(similarities[i]) / len(similarities[i])
        final_similarities.append(average_sim)
        txt_score = '{:.2f}'.format(final_similarities[i])
        score_X = int(x_offset + display_face_width/2)
        score_Y = int(y_offset + display_face_height + 80)
        cv2_putText(resultImg, txt_score, (score_X, score_Y), 80, (0,0,0), 2)
        #cv2.putText(resultImg, txt_score, (score_X, y_offset+img.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.75, player_color[i], 3, cv2.LINE_AA)
        
    #型変換
    resultImg = resultImg.astype('uint8')

    cv2.imshow('Camera 1',resultImg) #認識した顔の画像を表示

    return resultImg

font_dir = Path(__file__).resolve().parent / 'arial-unicode-ms.ttf'
def cv2_putText(img, text, org, fontScale, color, mode=0, fontFace = str(font_dir)):
    """日本語にも対応したputText

    Args:
        img (_type_): 元画像
        text (_type_): 追加するテキスト
        org (_type_): 描画座標
        fontFace (_type_): フォント
        fontScale (_type_): 文字の大きさ
        color (_type_): 文字の色
        mode (int, optional): 描画座標の種類の指定(左下or左上or中央). Defaults to 0.

    Returns:
        _type_: テキスト追加後の画像
    """
# cv2.putText()にないオリジナル引数「mode」　orgで指定した座標の基準
# 0（デフォ）＝cv2.putText()と同じく左下　1＝左上　2＝中央

    # テキスト描写域を取得
    fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (0,0)))
    text_w, text_h = dummy_draw.textsize(text, font=fontPIL)
    text_b = int(0.1 * text_h) # バグにより下にはみ出る分の対策

    # テキスト描写域の左上座標を取得（元画像の左上を原点とする）
    x, y = org
    offset_x = [0, 0, text_w//2]
    offset_y = [text_h, 0, (text_h+text_b)//2]
    x0 = x - offset_x[mode]
    y0 = y - offset_y[mode]
    img_h, img_w = img.shape[:2]

    # 画面外なら何もしない
    if not ((-text_w < x0 < img_w) and (-text_b-text_h < y0 < img_h)) :
        print ("out of bounds")
        return img

    # テキスト描写域の中で元画像がある領域の左上と右下（元画像の左上を原点とする）
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0+text_w, img_w), min(y0+text_h+text_b, img_h)

    # テキスト描写域と同サイズの黒画像を作り、それの全部もしくは一部に元画像を貼る
    text_area = np.full((text_h+text_b,text_w,3), (0,0,0), dtype=np.uint8)
    text_area[y1-y0:y2-y0, x1-x0:x2-x0] = img[y1:y2, x1:x2]

    # それをPIL化し、フォントを指定してテキストを描写する（色変換なし）
    imgPIL = Image.fromarray(text_area)
    draw = ImageDraw.Draw(imgPIL)
    draw.text(xy = (0, 0), text = text, fill = color, font = fontPIL)

    # PIL画像をOpenCV画像に戻す（色変換なし）
    text_area = np.array(imgPIL, dtype = np.uint8)

    # 元画像の該当エリアを、文字が描写されたものに更新する
    img[y1:y2, x1:x2] = text_area[y1-y0:y2-y0, x1-x0:x2-x0]

    return img