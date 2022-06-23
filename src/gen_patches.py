import os
# import warnings
# warnings.filterwarnings("ignore")

import numpy as np
import cv2
import uuid
from skimage import io, transform
from sklearn.model_selection import train_test_split

import tissueloc as tl
from ext.pyslide import pyramid, contour, patch
from ext.pycontour.cv2_transform import cv_cnt_to_np_arr
from ext.pycontour.poly_transform import np_arr_to_poly, poly_to_np_arr

from ext.pydaily.filesystem import overwrite_dir

def gen_samples(slides_dir, patch_level, patch_size, tumor_type, slide_list, dset, overlap_mode, patch_save_dir):
	# prepare saving directory
	patch_path = os.path.join(patch_save_dir, "patches", tumor_type)
	patch_img_dir = os.path.join(patch_path, dset, "imgs")
	if not os.path.exists(patch_img_dir):
		os.makedirs(patch_img_dir)
	patch_mask_dir = os.path.join(patch_path, dset, "masks")
	if not os.path.exists(patch_mask_dir):
		os.makedirs(patch_mask_dir)

	# processing slide one-by-one
	ttl_patch = 0
	slide_list.sort()
	for ind, ele in enumerate(slide_list):
		print("Processing {} {}/{}".format(ele, ind+1, len(slide_list)))
		cur_slide_path = os.path.join(slides_dir, ele+".svs")
		if not os.path.exists(cur_slide_path):
			cur_slide_path = os.path.join(slides_dir, ele+".SVS")

		# locate contours and generate batches based on tissue contours
		cnts, d_factor = tl.locate_tissue_cnts(
			cur_slide_path, max_img_size=2048, smooth_sigma=13,thresh_val=0.88, min_tissue_size=120000
		)
		select_level, select_factor = tl.select_slide_level(cur_slide_path, max_size=2048)
		cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)

		# scale contour to slide level 2
		wsi_head = pyramid.load_wsi_head(cur_slide_path)
		cnt_scale = select_factor / int(wsi_head.level_downsamples[patch_level])
		tissue_arr = cv_cnt_to_np_arr(cnts[0] * cnt_scale).astype(np.int32)
		# convert tissue_arr to convex if poly is not valid
		tissue_poly = np_arr_to_poly(tissue_arr)
		if tissue_poly.is_valid == False:
			tissue_arr = poly_to_np_arr(tissue_poly.convex_hull).astype(int)

		coors_arr = None
		if overlap_mode == "half_overlap":
			level_w, level_h = wsi_head.level_dimensions[patch_level]
			coors_arr = contour.contour_patch_splitting_half_overlap(
				tissue_arr, level_h, level_w, patch_size, inside_ratio=0.80
			)
		else:
			raise NotImplementedError("unknown overlapping mode")

		wsi_img = wsi_head.read_region((0, 0), patch_level, wsi_head.level_dimensions[patch_level])
		wsi_img = np.asarray(wsi_img)[:,:,:3]
		mask_path = os.path.join(slides_dir, "_".join([ele, tumor_type+".tif"]))
		mask_img = io.imread(mask_path)
		wsi_mask = (transform.resize(mask_img, wsi_img.shape[:2], order=0) * 255).astype(np.uint8) * 255

		# Create symbolic links for data splits
		dset_slides_dir = os.path.join(patch_save_dir, dset+"_slides")
		if not os.path.exists(dset_slides_dir):
			os.makedirs(dset_slides_dir)
		cur_slide_path_slink = os.path.join(dset_slides_dir, os.path.basename(cur_slide_path))
		if not os.path.islink(cur_slide_path_slink):
			os.symlink(cur_slide_path, cur_slide_path_slink)
		mask_path_slink = os.path.join(dset_slides_dir, os.path.basename(mask_path))
		if not os.path.islink(mask_path_slink):
			os.symlink(mask_path, mask_path_slink)

		for cur_arr in coors_arr:
			cur_h, cur_w = cur_arr[0], cur_arr[1]
			cur_patch = wsi_img[cur_h:cur_h+patch_size, cur_w:cur_w+patch_size]
			if cur_patch.shape[0] != patch_size or cur_patch.shape[1] != patch_size:
				continue
			cur_mask = wsi_mask[cur_h:cur_h+patch_size, cur_w:cur_w+patch_size]
			# background RGB (235, 210, 235) * [0.299, 0.587, 0.114]
			if patch.patch_bk_ratio(cur_patch, bk_thresh=0.864) > 0.88:
				continue

			if overlap_mode == "half_overlap" and tumor_type == "viable":
				pixel_ratio = np.sum(cur_mask > 0) * 1.0 / cur_mask.size
				if pixel_ratio < 0.05:
					continue

			patch_name = ele + "_" + str(uuid.uuid1())[:8]
			io.imsave(os.path.join(patch_img_dir, patch_name+".jpg"), cur_patch, check_contrast=False)
			io.imsave(os.path.join(patch_mask_dir, patch_name+".png"), cur_mask, check_contrast=False)
			ttl_patch += 1

	print("There are {} patches in total.".format(ttl_patch))

if __name__ == "__main__":
	# prepare train, validation and test slide list
	mask_dir = "../data/visualization/train/tissue_loc"
	slide_list = [os.path.splitext(ele)[0] for ele in os.listdir(mask_dir) if "png" in ele]
	train_slide_list, val_slide_list = train_test_split(slide_list, test_size=0.2, random_state=1234)
	val_slide_list, test_slide_list = train_test_split(val_slide_list, test_size=0.5, random_state=1234)
	# generate patches for segmentation model training
	slides_dir = "../data/dataset/train"
	patch_level, patch_size = 2, 512
	# tumor_type = "viable"
	tumor_types = ["viable", "whole"]
	patch_save_dir = "../data"
	for cur_type in tumor_types:
		print("Generating {} tumor patches.".format(cur_type))
		patch_modes = [
			(test_slide_list, "test", "half_overlap"),
			(val_slide_list, "val", "half_overlap"),
			(train_slide_list, "train", "half_overlap")
		]
		for mode in patch_modes:
			gen_samples(slides_dir, patch_level, patch_size, cur_type, mode[0], mode[1], overlap_mode=mode[2], patch_save_dir=patch_save_dir)