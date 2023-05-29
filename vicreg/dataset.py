import os
import random
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import find_neighbours

class PAIP2019DatasetVICReg(Dataset):
	def __init__(self, image_dir, mask_dir, transform=None):
		self.image_dir = image_dir
		self.mask_dir = mask_dir
		self.transform = transform
		self.images = os.listdir(image_dir)
		self.neighbour_dict = find_neighbours.create_patch_neighbour_dict(self.images)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		image_filename = self.images[index]
		image_path = os.path.join(self.image_dir, image_filename)
		image = np.array(Image.open(image_path).convert("RGB"))

		image_neighbours = self.neighbour_dict.get(image_filename)

		second_view_image_path = ""

		if len(image_neighbours) == 0:
			# No neighbours => use same patch as second view TODO: find better solution?
			second_view_image_path = image_path
		else:
			random_index = random.randint(0, len(image_neighbours) - 1)
			second_view_image_filename = image_neighbours[random_index]
			second_view_image_path = os.path.join(self.image_dir, second_view_image_filename)

		second_view_image = np.array(Image.open(second_view_image_path).convert("RGB"))

		if self.transform is not None:
			image, second_view_image = self.transform(image=image, second_view_image=second_view_image)

		return image, second_view_image

class PAIP2019DatasetWithRandomCrop(Dataset):
	def __init__(self, image_dir, mask_dir, transform=None, number_of_patches_to_reuse=0):
		self.image_dir = image_dir
		self.mask_dir = mask_dir
		self.transform = transform
		self.images = os.listdir(image_dir)
		self.neighbour_dict = find_neighbours.create_patch_neighbour_dict(self.images)
		self.number_of_patches_to_reuse = number_of_patches_to_reuse

	def __len__(self):
		return len(self.images + self.number_of_patches_to_reuse)

	def __getitem__(self, index):
		image_filename = self.images[index]
		image_path = os.path.join(self.image_dir, image_filename)
		image = np.array(Image.open(image_path).convert("RGB"))

		image_neighbours = self.neighbour_dict.get(image_filename)

		second_view_image_path = ""

		if len(image_neighbours) == 0:
			# No neighbours => use same patch as second view TODO: find better solution?
			second_view_image_path = image_path
		else:
			random_index = random.randint(0, len(image_neighbours) - 1)
			second_view_image_filename = image_neighbours[random_index]
			second_view_image_path = os.path.join(self.image_dir, second_view_image_filename)

		second_view_image = np.array(Image.open(second_view_image_path).convert("RGB"))

		if self.transform is not None:
			image, second_view_image = self.transform(image=image, second_view_image=second_view_image)

		return image, second_view_image