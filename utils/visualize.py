import torch
import os
import random
import itertools
import colorsys
import numpy as np
import torchvision
import torchvision.transforms.functional as TF
from skimage.measure import find_contours
import matplotlib.pyplot as plt
if "DISPLAY" not in os.environ:
    plt.switch_backend('agg')
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon


def calc_centroid(tensor):
    # Inputs Shape(1, 9 , 64, 64)
    # Return Shape(1, 9 ,2)
    tensor = tensor.float() + 1e-10
    n, l, h, w = tensor.shape
    indexs_y = torch.from_numpy(np.arange(h)).float().to(tensor.device)
    indexs_x = torch.from_numpy(np.arange(w)).float().to(tensor.device)
    center_y = tensor.sum(3) * indexs_y.view(1, 1, -1)
    center_y = center_y.sum(2, keepdim=True) / tensor.sum([2, 3]).view(n, l, 1)
    center_x = tensor.sum(2) * indexs_x.view(1, 1, -1)
    center_x = center_x.sum(2, keepdim=True) / tensor.sum([2, 3]).view(n, l, 1)
    output = torch.cat([center_y, center_x], 2)
    return output

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
        image: input image  (ndarray)
        mask : mask
        color : color of mask
        alpha : opacity of mask
    """
    temp_image = np.copy(image)

    if temp_image.dtype != np.uint8:
        temp_image = (temp_image * 255).astype(np.uint8)

    for c in range(3):
        temp_image[:, :, c] = np.where(mask != 0,
                                       temp_image[:, :, c] *
                                       (1 - alpha) + alpha * color[c] * 255,
                                       temp_image[:, :, c])
    return temp_image


def tensor_unnormalize(inp):

    # Input : Tensor array
    # 0utput : numpy array
    inp = inp.detach().cpu().numpy().transpose((1, 2, 0))
    #
    # mean = np.array([0.369, 0.314, 0.282])
    # std = np.array([0.282, 0.251, 0.238])
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def ndarray_unnormalize(inp):
    # Input : numpy array
    # 0utput : numpy array
    mean = np.array([0.369, 0.314, 0.282])
    std = np.array([0.282, 0.251, 0.238])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def tensor_imshow(inp, title=None):
    """Imshow for Tensor."""
    temp = inp.clone()
    temp = temp.detach().to('cpu')
    
    # temp = tensor_unnormalize(temp)
    plt.imshow(temp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def ndarray_imshow(inp, title=None):
    """Imshow for ndarray."""
    inp = ndarray_unnormalize(inp)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def imshow(inp, title=None):
    """Imshow ."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def get_masked_image(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, label_batch = \
        sample_batched['image'], sample_batched['labels']
    batch_size = len(images_batch)
    colors = random_colors(label_batch.shape[1])
    img_list = []
    for j in range(batch_size):
        # plt.subplot(1, batch_size, j + 1)
        image = images_batch[j]
        # Before imshow we have to unnormalize the image
        image_masked = tensor_unnormalize(image)
        for k in range(1, label_batch.shape[1]):
            color = colors[k]
            image_masked = apply_mask(image=image_masked,
                                      mask=label_batch[j][k], color=color, alpha=0.5)
        # plt.imshow(image_masked.clip(0, 255))
        image_masked = np.array(image_masked.clip(0, 255), dtype=np.uint8)
        img_list.append(TF.to_tensor(image_masked))
    out_img = torch.stack(img_list)
    return out_img


def show_mask(image, predict, title=None):
    binary_list = []
    predic_argm = predict.argmax(dim=1, keepdim=False)
    for i in range(predict.shape[1]):
        binary = (predic_argm == i)
        binary_list.append(binary)
    pred = torch.stack(binary_list, dim=1)
    pred = pred.detach().cpu().numpy()
    sample = {'image': image,
              'labels': pred}
    masked_image = get_masked_image(sample)
    out = torchvision.utils.make_grid(masked_image)
    imshow(out, title)

def save_mask_result(count, x, image, predict):
    binary_list = []
    predic_argm = predict.argmax(dim=1, keepdim=False)
    for i in range(predict.shape[1]):
        binary = (predic_argm == i)
        binary_list.append(binary)
    pred = torch.stack(binary_list, dim=1)
    pred = pred.detach().cpu().numpy()
    sample = {'image': image,
              'labels': pred}
    masked_image = get_masked_image(sample)
    out = torchvision.utils.make_grid(masked_image)
    out = out.detach().cpu().numpy().transpose((1, 2, 0))
    out = np.clip(out, 0, 1)
    plt.imshow(out)
    save_dir = os.path.join("/res/stage2_res/{}".format(x))
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, '{}.png'.format(count)))
