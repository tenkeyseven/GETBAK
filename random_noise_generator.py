import numpy as np
from torchvision.transforms import transforms
import torchvision

transform = transforms.Compose([
    transforms.ToTensor()
])

a = -1 + 2*np.random.rand(224,224,3)

a_clip = np.clip(a,-40/255,40/255)
print(a_clip.min(),a_clip.max())

# b = transform(a_clip)
b = a_clip
b = b+1
b = b/2
print(b.min(),b.max())

np.save('/home/nas928/ln/GETBAK/data/triggers/random_trigger_1.npy', a_clip)

torchvision.utils.save_image(transform(b),'t.png')

c = np.load('/home/nas928/ln/GETBAK/data/triggers/random_trigger_1.npy')

c = transform(c)

print(c.min(),c.max())

