import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class PAIP2019Dataset(Dataset):
	def __init__(self, image_dir, mask_dir, transform=None):
		self.image_dir = image_dir
		self.mask_dir = mask_dir
		self.transform = transform
		self.images = os.listdir(image_dir)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		img_path = os.path.join(self.image_dir, self.images[index])
		mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
		image = np.array(Image.open(img_path).convert("RGB"))
		mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.float32)
		mask[mask == 255.0] = 1.0 # might not be needed

		if self.transform is not None:
			image, mask = self.transform(image=image, mask=mask)
			# image = augmentations["image"]
			# mask = augmentations["mask"]
		return image, mask