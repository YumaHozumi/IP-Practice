import io
import numpy as np
from PIL import Image
import torch
import openpifpaf
import matplotlib.pyplot as plt

image = Image.open("./MAX76_gjmax20140531_TP_V.jpg")
image = image.resize((300, 256))
im_array = np.asarray(image)
im = Image.fromarray(im_array).convert("RGB")

predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
predictions, gt_anns, image_meta = predictor.pil_image(im)

annotation_painter = openpifpaf.show.AnnotationPainter()
with openpifpaf.show.image_canvas(im) as ax:
    annotation_painter.annotations(ax, predictions)
    plt.imshow(im)
    plt.show()
