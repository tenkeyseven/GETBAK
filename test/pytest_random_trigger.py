import numpy as np
import cv2
from torchvision.transforms import transforms

size = 80
trigger = np.random.uniform(0, 255, size * size * 3)

transform = transforms.Compose([
    transforms.ToTensor()
])


def save_image(img, fname):
	# img = img.data.numpy()
	img = np.transpose(img, (1, 2, 0))
	img = img[: , :, ::-1]
	cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])

a = np.reshape(trigger, (3,size,size))
print(a.shape)

save_image(a,'data/triggers/trigger_20.png')
