from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import cv2
from matplotlib.colors import Normalize as Norm
from torchvision import transforms
import numpy as np
from torchvision.transforms import functional as TF

from model_1 import FaceModel
import torch
from dataset import HelenDataset
from Helen_transform import Resize, ToPILImage, ToTensor, Normalize, RandomRotation, \
                                RandomResizedCrop

from visualize import random_colors, apply_mask, tensor_unnormalize, ndarray_imshow, imshow


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir = '/home/yinzi/Downloads/datas'


face_dataset = HelenDataset(txt_file='testing.txt',
                            root_dir=root_dir,
                            transform= transforms.Compose([
                                ToPILImage(),
                                Resize((64,64)),
                                ToTensor()
                                # Normalize(mean=[0.369, 0.314, 0.282],
                                #           std=[0.282, 0.251, 0.238])
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

def calc_F1(y, label):

    predictions = y
    fp = sum((predictions == 1) & (label == 0))
    fn = sum((predictions == 0) & (label == 1))
    tp = sum((predictions == 1) & (label == 1))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * precision * recall / (precision + recall)


# Helper function to show a batch
def show_label_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, label_batch = \
        sample_batched['image'], sample_batched['labels']
    batch_size = len(images_batch)
    colors = random_colors(label_batch.shape[1])
    for j in range(batch_size):
        plt.subplot(1, batch_size, j + 1)
        image = images_batch[j]
        # Before imshow we have to unnormalize the image
        image_masked = tensor_unnormalize(image)
        img_list = []
        for k in range(1, label_batch.shape[1]):
            color = colors[k]
            image_masked = apply_mask(image=image_masked,
                                      mask=label_batch[j][k], color=color, alpha=0.5)
        plt.imshow(image_masked.clip(0, 255))
        img_list.append(image_masked)
    plt.show()
    return img_list



model = FaceModel().to(device)
state = torch.load('stage1.pth.tar',map_location=device)
state = state['model']
model.load_state_dict(state)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['labels'].size())
    image, labels = sample_batched['image'], sample_batched['labels']

    image = image.to(device)
    pred = model(image)
    # np_a = np_a.detach().cpu().numpy()
    # Change the predict-label to one-hot
    pred = torch.softmax(pred, 1)

    binary_list = []
    image_argm = image.argmax(dim=1, keepdim=False)
    predic_argm = pred.argmax(dim=1, keepdim=False)
    for i in range(pred.shape[1]):
        binary = (predic_argm == i)
        binary_list.append(binary)
    pred = torch.stack(binary_list, dim=1)
    pred = pred.detach().cpu().numpy()

    sample = {'image':image ,
              'labels': pred}
    show_label_batch(sample)

