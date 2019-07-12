from dataset import HelenDataset
from torch.utils.data import DataLoader

from torchvision import transforms

from Helen_transform import Resize, ToPILImage, ToTensor, Normalize, RandomRotation, \
                                RandomResizedCrop, LabelsToOneHot

import numpy as np

root_dir = '/home/yinzi/Downloads/datas'

dset = HelenDataset(txt_file='exemplars.txt',
                    root_dir=root_dir,
                    transform= transforms.Compose([
                        ToPILImage(),
                        Resize((64, 64)),
                        ToTensor()
                    ])
                    )

dataloader = DataLoader(dset, batch_size=2330, shuffle=False, num_workers=4)

batches = iter(dataloader).next()
# images.shape = ( 32, 3, 80, 80)
images , labels = batches['image'], batches['labels']

numpy_images = images.numpy()

per_image_mean = np.mean(numpy_images, axis=(2,3)) #Shape (32,3)
per_image_std = np.std(numpy_images, axis=(2,3)) #Shape (32,3)

pop_channel_mean = np.mean(per_image_mean, axis=0) # Shape (3,)
pop_channel_std = np.mean(per_image_std, axis=0) # Shape (3,)

print(pop_channel_mean)
print(pop_channel_std)

# 2330
# mean:[0.368689   0.3144578  0.28216377]
# std:[0.28209248 0.2509268  0.23772953]
