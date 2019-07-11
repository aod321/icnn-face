from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from dataset import HelenDataset,Rescale,ToTensor
from visualize import random_colors,apply_mask
from torchvision import transforms, utils
import numpy as np

face_dataset = HelenDataset(txt_file='exemplars.txt',
                            root_dir='/home/yinzi/Downloads/datas',
                            transform=transforms.Compose([Rescale((480,480)),
                                                          ToTensor()])
                            )


dataloader = DataLoader(face_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


# Helper function to show a batch
def show_label_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, label_batch =\
        sample_batched['image'], sample_batched['labels']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    colors = random_colors(8)
    print(colors)
    # example output
    # colors =[(0.5, 0.0, 1.0), (0.0, 0.25, 1.0), (0.5, 1.0, 0.0),
    # (1.0, 0.75, 0.0), (1.0, 0.0, 0.75), (0.0, 1.0, 0.25), (0.0, 1.0, 1.0), (1.0, 0.0, 0.0)]

    plt.figure()
    for j in range(batch_size):
        plt.subplot(1, batch_size, j + 1)
        image = images_batch[j]
        image_masked = image.numpy().transpose((1, 2, 0))
        for k in range(8):
            color = colors[k]
            image_masked = apply_mask(image=image_masked,
                                      mask=label_batch[j][k], color=color,alpha=0.8)

        plt.imshow((image_masked * 255).astype(np.uint8))
    plt.title('Batch from dataloader')


for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['labels'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_label_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break



#
#
#
#
# def show_four():
#     fig = plt.figure()
#
#     colors = random_colors(8)
#     for i in range(len(face_dataset)):
#         sample = face_dataset[i]
#
#         print(i, sample['image'].shape, sample['labels'].shape)
#
#         ax = plt.subplot(2, 2, i + 1)
#         plt.tight_layout()
#         ax.set_title('Sample #{}'.format(i))
#         ax.axis('off')
#         image_mask = sample['image']
#         for j in range(8):
#             color = colors[j]
#             image_mask = apply_mask(image=image_mask, mask=sample['labels'][j], color=color)
#
#         plt.imshow(image_mask)
#
#         if i == 3:
#             plt.show()
#             break
#
