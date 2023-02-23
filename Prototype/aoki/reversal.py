import cv2

img = cv2.imread('MAX76_gjmax20140531_TP_V.jpg')

# 左右反転させるコード
img_flip_lr = cv2.flip(img, 1)
cv2.imwrite('MAX76_reverse.jpg', img_flip_lr)