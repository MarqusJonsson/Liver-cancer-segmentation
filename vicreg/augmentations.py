# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class TrainTransform(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(), 
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.transform_second_view = transforms.Compose(
            [
                transforms.ToPILImage(), 
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, image, second_view_image):
        transformed_image = self.transform(image)
        transformed_second_view_image = self.transform_second_view(second_view_image)
        return transformed_image, transformed_second_view_image
