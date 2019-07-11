import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt


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
        image: input iamge
        mask : mask
        color : color of mask
        alpha : opacity of mask
    """
    temp_image = np.copy(image)
    for c in range(3):
        temp_image[:, :, c] = np.where(mask != 0,
                                       temp_image[:, :, c] *
                                       (1 - alpha) + alpha * color[c] * 255,
                                       temp_image[:, :, c])
    return temp_image


