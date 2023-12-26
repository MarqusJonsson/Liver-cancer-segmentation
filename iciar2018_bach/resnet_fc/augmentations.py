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

    def __call__(self, image):
        transformed_image = self.transform(image)
        return transformed_image
