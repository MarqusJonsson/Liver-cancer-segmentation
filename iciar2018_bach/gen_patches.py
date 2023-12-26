import os
import cv2
import tifffile
import openslide
import numpy as np
from tqdm import trange
from sklearn.model_selection import train_test_split
from PIL import Image

def gen_patches(photos_dir, patch_save_dir, photos_list, dset, patch_size, patch_overlap, class_name):
	if not os.path.exists(patch_save_dir):
		os.makedirs(patch_save_dir)

	photos_list.sort()

	for idx in trange(0, len(photos_list)):
		cur_photo_path = os.path.join(photos_dir, class_name, photos_list[idx]) # slide path without file extension
		
		img = Image.open(cur_photo_path)
# 		wsi = openslide.OpenSlide(cur_slide_path + ".svs")
# 		wsi_w, wsi_h = wsi.dimensions
		width, height = img.size
		img = np.array(img)

		patch_dir = os.path.join(patch_save_dir, dset)
		if not os.path.exists(patch_dir):
			os.makedirs(patch_dir)

		step = int(patch_size * (1.0 - patch_overlap))
		if dset == "test":
			# Don't use overlapping patches for test set
			step = patch_size

		for i in range(0, height - patch_size, step):
			for j in range(0, width - patch_size, step):
				patch = img[i:i + patch_size, j:j + patch_size]
				patch = cv2.cvtColor(patch,cv2.COLOR_RGB2BGR)
				cv2.imwrite(os.path.join(patch_dir, photos_list[idx] + f"_{class_name}_{idx}_{i}_{j}.jpg"), patch)

if __name__ == "__main__":
	photos_dir = "../../ICIAR2018/ICIAR2018_BACH_Challenge/Photos"
	photos_dict = {os.path.splitext(entry)[0] : {"all": [], "train": [], "val": [], "test": []} for entry in os.listdir(photos_dir) if os.path.isdir(os.path.join(photos_dir, entry))}
	for class_name in photos_dict:
		for photo in os.listdir(os.path.join(photos_dir, class_name)):
			if os.path.splitext(photo)[-1] != ".tif":
				continue
			photos_dict[class_name]["all"].append(photo)

	for class_name in photos_dict:
		train_list, val_test_list = train_test_split(photos_dict[class_name]["all"], test_size=0.2, random_state=1234)
		val_list, test_list = train_test_split(val_test_list, test_size=0.5, random_state=1234)
		photos_dict[class_name]["train"] = train_list
		photos_dict[class_name]["val"] = val_list
		photos_dict[class_name]["test"] = test_list
	
	patch_size = 512
	patch_overlap = 0.8
	patch_save_dir = f"../data/patches/iciar2018_tr80_v10_te10_ps_{patch_size}_po_{patch_overlap}"

	for class_name in photos_dict:
		patch_modes = [
			(photos_dict[class_name]["val"], "val"),
			(photos_dict[class_name]["train"], "train"),
			(photos_dict[class_name]["test"], "test")
		]
		for mode in patch_modes:
			print(mode[1], "starting")
			gen_patches(
				photos_dir,
				patch_save_dir,
				photos_list=mode[0],
				dset=mode[1],
				patch_size=patch_size,
				patch_overlap=patch_overlap,
				class_name=class_name
			)
			print(mode[1], class_name, "complete")