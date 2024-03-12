from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from filter import FastGuidedFilter


def get_image(path):
    """
    Reads and returns RGB image, (1,3,H,W).
    """
    image = torch.from_numpy(np.array(Image.open(path))).float()
    image = image / torch.max(image)
    image = torch.movedim(image, -1, 0).unsqueeze(0).cuda()
    return image


def get_v_component(img_hsv):
    """
    Assumes (1,3,H,W) HSV image.
    """
    return img_hsv[:,-1].unsqueeze(0)


def replace_v_component(img_hsv, v_new):
    """
    Replaces the V component of a HSV image (1,3,H,W).
    """
    img_hsv[:,-1] = v_new
    return img_hsv


def interpolate_image(img, H, W):
    """
    Reshapes the image based on new resolution.
    """
    return F.interpolate(img, size=(H,W))


def get_coords(H, W):
    """
    Creates a coordinates grid for INF.
    """
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W)))
    coords = torch.from_numpy(coords).float().cuda()
    return coords


def get_patches(img, KERNEL_SIZE):
    """
    Creates a tensor where the channel contains patch information.
    """
    kernel = torch.zeros((KERNEL_SIZE ** 2, 1, KERNEL_SIZE, KERNEL_SIZE)).cuda()

    for i in range(KERNEL_SIZE):
        for j in range(KERNEL_SIZE):
            kernel[int(torch.sum(kernel).item()),0,i,j] = 1

    pad = nn.ReflectionPad2d(KERNEL_SIZE//2)
    im_padded = pad(img)

    extracted = torch.nn.functional.conv2d(im_padded, kernel, padding=0).squeeze(0)

    return torch.movedim(extracted, 0, -1)


def filter_up(x_lr, y_lr, x_hr, r=1):
    """
    Applies the guided filter to upscale the predicted image.
    """
    guided_filter = FastGuidedFilter(r=r)
    y_hr = guided_filter(x_lr, y_lr, x_hr)
    y_hr = torch.clip(y_hr, 0, 1)
    return y_hr