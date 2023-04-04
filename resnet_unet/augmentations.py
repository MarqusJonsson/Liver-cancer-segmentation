import torchvision.transforms as transforms

class TrainTransform(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                # transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )
        self.transform_second_view = transforms.Compose(
            [
                # transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, image, mask):
        transformed_image = self.transform(image)
        transformed_mask = self.transform_second_view(mask)
        return transformed_image, transformed_mask
