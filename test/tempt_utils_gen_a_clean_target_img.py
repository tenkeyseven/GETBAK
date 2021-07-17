import numpy
from PIL import Image
import torch
from torchvision.transforms import transforms
import cv2
import numpy as np

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

def save_image(img, fname):
	img = img.data.numpy()
	img = np.transpose(img, (1, 2, 0))
	img = img[: , :, ::-1]
	cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])

path = './datasets/imagenette/imagenette2/train'

im = Image.open(path + '/n03425413/ILSVRC2012_val_00017469.JPEG').convert('RGB')

im = data_transform(im)

save_image(im,'./data/clean_target_images/ct_img_7_001.png')