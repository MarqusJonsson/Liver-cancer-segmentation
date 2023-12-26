import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class ICIAR2018Dataset(Dataset):
	def __init__(self, image_dir, transform=None):
		self.image_dir = image_dir
		self.transform = transform
		self.images = os.listdir(image_dir)
		self.possible_labels = ["Normal", "Benign", "InSitu", "Invasive"]

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		img_name = self.images[index]
		img_path = os.path.join(self.image_dir, img_name)
		image = np.array(Image.open(img_path).convert("RGB"))
		label = ""
		for possible_label in self.possible_labels:
			if possible_label in img_name:
				label = possible_label
				break

		if self.transform is not None:
			image = self.transform(image=image)

		return image, self._create_one_hot_vector(label), img_name


	# Create a one-hot encoded vector for a given string
	def _create_one_hot_vector(self, input_string):
		# Create a one-hot encoded vector with the length of the possible strings
		one_hot_vector = np.zeros(len(self.possible_labels))

		# Set the index corresponding to the input string to 1
		one_hot_vector[self.possible_labels.index(input_string)] = 1

		return one_hot_vector