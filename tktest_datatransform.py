from utils.transforms_utils import data_transform, transform_unNormalize, trigger_transform
from PIL import Image
import torchvision

im = Image.open('/home/nas928/ln/GETBAK/datasets/imagenette/imagenette_poisoned/train/n03425413/ILSVRC2012_val_00019667.JPEG').convert('RGB')

im = data_transform(im)

torchvision.utils.save_image(transform_unNormalize(im),'tempt_output/t.png')