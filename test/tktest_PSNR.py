# PSNR高于40dB说明图像质量极好（即非常接近原始图像），在30—40dB通常表示图像质量是好的（即失真可以察觉但可以接受），在20—30dB说明图像质量差；最后，但PSNR低于20dB图像不可接受
import numpy as np
import math
from PIL import Image

def PSNR(img1, img2):
    mse = np.mean((img1 - img2)** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

img1 = Image.open("./metric_test_image/image1.jpg")
img2 = Image.open("./metric_test_image/image2.jpg")

img1_array = np.array(img1)
img2_array = np.array(img2)

print("The PSNR between image1 and image2 is: %.3f" % PSNR(img1_array,img2_array))
