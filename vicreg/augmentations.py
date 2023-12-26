# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from CustomTransform import LogarithmicMappingTransform
from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
        self.transform = A.Compose(
            [
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.3),
                A.Flip(always_apply=False, p=0.3),
                A.Rotate(always_apply=False, p=0.3),
                A.Affine(translate_percent=0.05, always_apply=False, p=0.3),
                A.RandomResizedCrop(height=224, width=224, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                LogarithmicMappingTransform(always_apply=False, p=0.45),
                ToTensorV2(),
            ]
        )
        self.transform_second_view = A.Compose(
            [
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.3),
                A.Flip(always_apply=False, p=0.3),
                A.Rotate(always_apply=False, p=0.3),
                A.Affine(translate_percent=0.05, always_apply=False, p=0.3),
                A.RandomResizedCrop(height=224, width=224, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                LogarithmicMappingTransform(always_apply=False, p=0.45),
                ToTensorV2(),
            ]
        )

    def __call__(self, image, second_view_image):
        transformed_image = self.transform(image=image)
        transformed_second_view_image = self.transform_second_view(image=second_view_image)
        return transformed_image["image"], transformed_second_view_image["image"]
