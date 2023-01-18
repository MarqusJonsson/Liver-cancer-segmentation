import os
import cv2
import tifffile
import openslide
import numpy as np
from tqdm import trange
from sklearn.model_selection import train_test_split

def gen_patches(slides_dir, slide_list, dset, patch_size, patch_save_dir, tissue_masks_dir, min_tissue=None, patch_overlap=0.5):
	if not os.path.exists(patch_save_dir):
		os.makedirs(patch_save_dir)

	slide_list.sort()

	if min_tissue is None:
		min_tissue = patch_size * patch_size * 0.5 # minimum 0.5 => 50% or more tissue on every patch

	for idx in trange(0, len(slide_list)):
		cur_slide_path = os.path.join(slides_dir, slide_list[idx]) # slide path without file extension
		wsi = openslide.OpenSlide(cur_slide_path + ".svs")
		wsi_w, wsi_h = wsi.dimensions

		tissue_mask = tifffile.imread(os.path.join(tissue_masks_dir, slide_list[idx]) + f"_tissue.tif")
		whole_mask = tifffile.imread(cur_slide_path + "_whole.tif")
		viable_mask = tifffile.imread(cur_slide_path + "_viable.tif")

		tissue_patch_dir = os.path.join(patch_save_dir, dset, "tissue")
		tissue_mask_patch_dir = os.path.join(patch_save_dir, dset, "tissue_mask")
		whole_patch_dir = os.path.join(patch_save_dir, dset, "whole")
		viable_patch_dir = os.path.join(patch_save_dir, dset, "viable")
		if not os.path.exists(tissue_patch_dir):
			os.makedirs(tissue_patch_dir)
		if not os.path.exists(tissue_mask_patch_dir):
			os.makedirs(tissue_mask_patch_dir)
		if not os.path.exists(whole_patch_dir):
			os.makedirs(whole_patch_dir)
		if not os.path.exists(viable_patch_dir):
			os.makedirs(viable_patch_dir)
		step = int(patch_size * (1.0 - patch_overlap))
		for i in range(0, wsi_h - patch_size, step):
			for j in range(0, wsi_w - patch_size, step):
				tissue_mask_patch = tissue_mask[i:i + patch_size, j:j + patch_size]
				tissue_mask_sum = tissue_mask_patch.sum() / 255
				if tissue_mask_sum < min_tissue:
					continue
				# if tissue_mask[i:i+window_size,j:j+window_size].sum()<400000:
				# 	continue
				tissue_patch = wsi.read_region((j, i), 0, (patch_size, patch_size)).convert("RGB")
				tissue_patch = np.array(tissue_patch)
				# patch=cv2.cvtColor(patch,cv2.COLOR_RGB2BGR)
				whole_mask_patch = whole_mask[i:i + patch_size, j:j + patch_size]
				viable_mask_patch = viable_mask[i:i + patch_size, j:j + patch_size]
				cv2.imwrite(os.path.join(tissue_patch_dir, slide_list[idx] + f"_{idx}_{i}_{j}.jpg"), tissue_patch)
				cv2.imwrite(os.path.join(tissue_mask_patch_dir, slide_list[idx] + f"_{idx}_{i}_{j}.png"), tissue_mask_patch)
				cv2.imwrite(os.path.join(whole_patch_dir, slide_list[idx] + f"_{idx}_{i}_{j}.png"), whole_mask_patch)
				cv2.imwrite(os.path.join(viable_patch_dir, slide_list[idx] + f"_{idx}_{i}_{j}.png"), viable_mask_patch)

if __name__ == "__main__":
	slides_dir = "../data/dataset/train"
	slide_list = [os.path.splitext(ele)[0] for ele in os.listdir(slides_dir) if "svs" in ele]

	train_slide_list, val_slide_list = train_test_split(slide_list, test_size=0.2, random_state=1234)
	val_slide_list, test_slide_list = train_test_split(val_slide_list, test_size=0.5, random_state=1234)

	patch_size = 1024
	patch_overlap = 0.5
	min_tissue_const = 0.8 # minimum 0.8 => 80% or more tissue on every patch
	min_tissue = patch_size * patch_size * min_tissue_const
	patch_save_dir = f"../data/patches/ps_{patch_size}_po_{patch_overlap}_mt_{min_tissue_const}_temp"


	tissue_masks_dir = "../data/tissue_masks"

	patch_modes = [
		(test_slide_list, "test"),
		(val_slide_list, "val"),
		(train_slide_list, "train")
	]
	for mode in patch_modes:
		gen_patches(
			slides_dir,
			slide_list=mode[0],
			dset=mode[1],
			patch_size=patch_size,
			patch_save_dir=patch_save_dir,
			tissue_masks_dir=tissue_masks_dir,
			min_tissue=min_tissue,
			patch_overlap=patch_overlap,
		)