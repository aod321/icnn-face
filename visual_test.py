from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as Norm
from torchvision import transforms
import numpy as np
from torchvision.transforms import functional as TF

from dataset import HelenDataset

from Helen_transform import Resize, ToPILImage, ToTensor, Normalize, RandomRotation, \
                                RandomResizedCrop

from visualize import random_colors, apply_mask, tensor_unnormalize, ndarray_imshow, imshow

# root_dir = "/data1/yinzi/datas"
root_dir = '/home/yinzi/Downloads/datas'

face_dataset = HelenDataset(txt_file='exemplars.txt',
                            root_dir=root_dir,
                            transform= transforms.Compose([
                                ToPILImage(),
                                RandomRotation(15),
                                RandomResizedCrop((255, 255), scale=(0.9, 1.1)),
                                Resize((64, 64)),
                                ToTensor(),
                                Normalize(mean=[0.369, 0.314, 0.282],
                                          std=[0.282, 0.251, 0.238])
                            ])
                            )


dataloader = DataLoader(face_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


def imshow_with_label(sample):
    img, labels = sample['image'], sample['labels']
    colors = random_colors(8)
    image_masked = tensor_unnormalize(img)
    for k in range(8):
        color = colors[k]
        image_masked = apply_mask(image=image_masked,
                                  mask=labels[k], color=color, alpha=0.5)
    imshow(image_masked)

# Helper function to show a batch
def show_label_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, label_batch =\
        sample_batched['image'], sample_batched['labels']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    colors = random_colors(8)
    # print(colors)
    # example output
    # colors =[(0.5, 0.0, 1.0), (0.0, 0.25, 1.0), (0.5, 1.0, 0.0),
    # (1.0, 0.75, 0.0), (1.0, 0.0, 0.75), (0.0, 1.0, 0.25), (0.0, 1.0, 1.0), (1.0, 0.0, 0.0)]
    for j in range(batch_size):
        plt.subplot(2, 2, j + 1)
        image = images_batch[j]
        # Before imshow we have to unnormalize the image
        image_masked = tensor_unnormalize(image)
        for k in range(8):
            color = colors[k]
            image_masked = apply_mask(image=image_masked,
                                      mask=label_batch[j][k], color=color, alpha=0.5)

        plt.imshow(image_masked.clip(0, 255))

    plt.show()


for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['labels'].size())


    show_label_batch(sample_batched)


    if i_batch == 4:

        break
