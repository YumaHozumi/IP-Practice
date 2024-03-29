import cv2
import openpifpaf
import numpy as np
from typing import List, Tuple
from .functions import get_draw_info, create_connected
from .draw_function import draw_peopleNum
from .display_functions import display_registered_playeres
from .settings import SCALE_UP, Result_X, Result_Y 
from .area_settings import X_LIMIT_START, Y_LIMIT_START, X_LIMIT_END, Y_LIMIT_END, face_width, face_height
from .area_settings import Window_width, Window_height, display_face_width, display_face_height

def register(capture: cv2.VideoCapture, predictor: openpifpaf.predictor.Predictor) -> List[np.ndarray]:
    """プレイヤーを登録する

    Args:
        capture (cv2.VideoCapture): キャプチャー
        predictor (openpifpaf.predictor.Predictor): 関節点推定モデル

    Returns:
        List[np.ndarray]: 登録した顔画像のリスト
    """

    #プレイヤー登録の指示を表示
    #指示画像の用意
    register_inst = cv2.imread('./modules/pictures/Register_instruction.bmp', cv2.IMREAD_COLOR )
    register_inst = cv2.resize(register_inst, (Window_width, Window_height))

    register_finished: bool = False #登録が完了したかどうか

    #プレイヤーの登録が終了するまで登録作業を繰り返す
    while not register_finished:
        #指示を表示
        cv2.imshow('Camera 1', register_inst)
        while True:
            if cv2.waitKey(10) == 0x0d: 
                break
            
        #登録する顔のリストを得る
        face_Imgs, display_face_Imgs = capture_registerArea(capture, predictor)
        #登録結果の描画(一応登録者一覧画面をもらってるが、今のところ再利用する予定なし)
        result = display_registered_playeres(display_face_Imgs)

        while True: 
            #キーボード入力を受け取る
            key = cv2.waitKey(10)
            # Enterキーを押すと登録完了
            if key == 0x0d:
                print('Enter pressed. Register finished...')
                register_finished = True
                break
            #Deleteを押すと登録をやり直し
            elif key == 127:
                break

    finish_message = cv2.imread('./modules/pictures/Register_finish.bmp', cv2.IMREAD_COLOR )
    finish_message = cv2.resize(finish_message, (Window_width, Window_height))
    
    return face_Imgs, display_face_Imgs

def capture_registerArea(capture: cv2.VideoCapture, predictor: openpifpaf.predictor.Predictor) -> List[np.ndarray]:
    """登録領域を撮影し、顔画像を抽出する

    Args:
        capture (cv2.VideoCapture): キャプチャー
        predictor (openpifpaf.predictor.Predictor): 関節点推定モデル

    Returns:
        List[np.ndarray]: 顔画像のリスト
    """

    while capture.isOpened():
        #success: 画像の取得が成功したか
        #frame: RGBの値を持っている3次元の配列データ ex) サイズ (480, 640, 3) 高さ、幅、色チャネル
        read_video: Tuple[bool, np.ndarray] = capture.read()
        success, frame = read_video
        # print("frame1 =",frame)

        if not success :
            print( "frame is None" )
            break
        
        
        #登録をおこなう領域を指定
        register_frame = frame[Y_LIMIT_START:Y_LIMIT_END, X_LIMIT_START:X_LIMIT_END]
        #登録領域で認識を行う(ここは人数だけ分かればいい)
        resize_frame: np.ndarray = cv2.resize(register_frame, dsize=None, fx=(1.0 / SCALE_UP), fy=(1.0 / SCALE_UP))
        predictions, gt_anns, meta = predictor.numpy_image(resize_frame)
        #predictions: 関節座標
        #インデックス: 関節座標点
        
        annotated_image = frame.copy()

        #認識領域に人が映ってないときにもカメラ映像を出すように
        if len(predictions) == 0: 
            annotated_image = cv2.rectangle(annotated_image, (X_LIMIT_START, Y_LIMIT_START), (X_LIMIT_END, Y_LIMIT_END), (0,255,0), thickness=2)
            output_image = cv2.flip(annotated_image, 1)
            display_frame = cv2.resize(output_image, (Window_width, Window_height))
            cv2.imshow('Camera 1',display_frame)
            
            # ESCキーを押すと終了
            if cv2.waitKey(10) == 0x1b:
                print('ESC pressed. Exiting ...')
                break
            
            continue

        annotated_image = cv2.rectangle(annotated_image, (X_LIMIT_START, Y_LIMIT_START), (X_LIMIT_END, Y_LIMIT_END), (0,255,0), thickness=2)
        annotated_image = cv2.flip(annotated_image, 1)
        
        #人数の描画
        registable_label = registerable_check(predictions)
        peopleNumber = np.sum(registable_label)
        annotated_image = draw_peopleNum(annotated_image, peopleNumber)

        display_frame = cv2.resize(annotated_image, (Window_width, Window_height))
        cv2.imshow('Camera 1',display_frame)
        #cv2.moveWindow("Camera 1", 200,40)

        # Enterキーを押したら画像の読み込みを終了
        key = cv2.waitKey(10)
        if (key == 0x0d and peopleNumber > 1 and peopleNumber < 5):
            print('Enter pressed. Saving ...')
            break
        else:
            key = 0

    #Enter押下時の画像から顔領域を抽出し、表示する
    face_Imgs, display_faceImgs = regist_faceImg(register_frame, predictions, registable_label)
    return face_Imgs, display_faceImgs


def registerable_check(landmarks: np.ndarray) -> np.ndarray:
    """顔画像登録が可能な人の関節点情報のみ抽出する

    Args:
        landmarks (np.ndarray): 関節点座標のリスト

    Returns:
        np.ndarray: 登録可能な人を表すラベル
    """

    registable_list = np.ones(len(landmarks)) #登録可能な人を表すラベル

    #登録可能リストを作成する
    for num in range(len(landmarks)):
        parts = landmarks[num]
        #左目と左耳のいずれかが存在し、かつ右目と右耳のいずれか存在する場合登録可能とする
        if(((parts.data[1][2]==0)and(parts.data[3][2]==0)) or ((parts.data[2][2]==0)and(parts.data[4][2]==0))):
            registable_list[num] *= 0
    
    #print(registable_list)
    #print("--------------")
    return registable_list

def regist_faceImg(register_frame: np.ndarray, landmarks: np.ndarray, label: np.ndarray) -> List[np.ndarray]:
    """顔画像を登録する

    Args:
        register_frame (np.ndarray): 登録領域の画像
        landmarks (np.ndarray): 関節点座標のリスト
        label (np.ndarray): 顔画像を登録できるかのラベル

    Returns:
        List[np.ndarray]: 登録する顔画像のリスト
    """

    faceImgs: List[np.ndarray] = []
    display_faceImgs: List[np.ndarray] = []

    for num in range(len(landmarks)):
        parts = landmarks[num]

        #顔画像領域を計算
        start_X = 0
        start_Y = 0
        end_X = face_width
        end_Y = face_height
        if(label[num] == 1):
            if(parts.data[4][2]>0):
                start_X = parts.data[4][0] * SCALE_UP - 20 
                start_Y = parts.data[4][1] * SCALE_UP - 100
                end_Y =  parts.data[4][1] * SCALE_UP + 100
            elif(parts.data[2][2]>0):
                start_X = parts.data[2][0] * SCALE_UP - 60 
                start_Y = parts.data[2][1] * SCALE_UP - 100
                end_Y =  parts.data[2][1] * SCALE_UP + 100

            if(parts.data[3][2]>0):
                end_X = parts.data[3][0] * SCALE_UP + 20
            elif(parts.data[1][2]>0):
                end_X = parts.data[1][0] * SCALE_UP + 60

            #領域外になった時の処理(正しいかどうかはちょっと・・・)
            if(start_X < 0): start_X = 0
            if(start_Y < 0): start_Y = 0
            if(end_X > (X_LIMIT_END - X_LIMIT_START)): end_X = X_LIMIT_END - X_LIMIT_START
            if(end_Y > (Y_LIMIT_END - Y_LIMIT_START)): end_Y = Y_LIMIT_END - Y_LIMIT_START
    
            #顔画像領域を抽出
            face_frame = register_frame[int(start_Y):int(end_Y), int(start_X):int(end_X)]
            #print(face_frame.shape)

            #サイズ調整
            face_frame = cv2.resize(face_frame, (face_width, face_height))
            #print(face_frame.shape)
            display_face_frame = cv2.resize(face_frame, (display_face_width, display_face_height))

            faceImgs.append(face_frame)
            display_faceImgs.append(display_face_frame)

    #print(type(faceImgs))
    #print(type(faceImgs[0]))
    return faceImgs, display_faceImgs
