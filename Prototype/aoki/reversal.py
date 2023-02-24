from PIL import Image
import cv2
import numpy as np

if __name__ == "__main__":
    img = np.asarray(Image.open('MAX76_gjmax20140531_TP_V.jpg'), dtype=np.uint8)
    img = cv2.flip(img, 1)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("MAX76_reverse.jpg", im_rgb)
#    plt.imshow(img)
#    plt.xticks(color="None")
#    plt.yticks(color="None")
#    plt.tick_params(length=0)
#    plt.savefig('MAX76_reverse.jpg')