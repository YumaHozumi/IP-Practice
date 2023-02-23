from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math

if __name__ == "__main__":
    img = np.asarray(Image.open('MAX76_gjmax20140531_TP_V.jpg'), dtype=np.uint8)
    img = img[:, ::-1, :]
    plt.imshow(img)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    plt.savefig('MAX76_reverse.jpg')