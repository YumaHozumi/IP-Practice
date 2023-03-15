import cv2
import openpifpaf
import numpy as np
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
from functions import get_draw_info, create_connected
from settings import SCALE_UP, Result_X, Result_Y 
from area_settings import X_LIMIT_START, Y_LIMIT_START, X_LIMIT_END, Y_LIMIT_END, face_width, face_height, Window_width, Window_height

whiteboard = 255 * np.ones([Window_height, Window_width, 3])

#プレイヤーを識別する色(B,G,R)
player_color: List = [(0, 0, 255),(255, 0, 0),(0, 255, 0),(0, 165, 255)]

def display_registered_playeres(face_Imgs: List[np.array]) -> np.ndarray:
    """登録されたプレイヤー一覧を表示する

    Args:
        face_Imgs (List[np.array]): 顔画像のリスト

    Returns:
        np.ndarray: 一覧を表示している画像
    """
    playeresImg = whiteboard.copy() #背景の設定

    #画面の説明の表示
    cv2_putText(playeresImg, 'プレイヤー一覧', (20, 80), 80, (0,0,0))
    cv2_putText(playeresImg, '　OK!　  > Enter', (int(Window_width * 0.7), Window_height - 50), 40, (0,0,0))
    cv2_putText(playeresImg, 'やり直す > Delete', (int(Window_width * 0.7), Window_height - 10), 40, (0,0,0))
    
    #登録結果表示画面の作成
    people_num = len(face_Imgs)
    for i in range(people_num):
        img = face_Imgs[i] #顔画像

        #描画領域の指定
        separate_width = Window_width / (people_num + 1)
        x_offset=int((i+1)*separate_width - face_width/2)
        y_offset=int(Window_height/2 - face_height/2)
        playeresImg[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img.copy()
        txt = "Player" + str(int(i + 1))
        cv2.putText(playeresImg, txt, (x_offset, y_offset - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.75, player_color[i], 3, cv2.LINE_AA)
        
    #型変換
    playeresImg = playeresImg.astype('uint8')

    cv2.imshow('Camera 1',playeresImg) #認識した顔の画像を表示

    return playeresImg

def cv2_putText(img, text, org, fontScale, color, mode=0, fontFace = "./arial-unicode-ms.ttf"):
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