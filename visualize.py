import torch
import os
import random
import itertools
import colorsys
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
if "DISPLAY" not in os.environ:
    plt.switch_backend('agg')
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon



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
    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.369, 0.314, 0.282])
    std = np.array([0.282, 0.251, 0.238])
    inp = std * inp + mean
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
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated