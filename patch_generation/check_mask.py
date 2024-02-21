import os
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt

# from pydaily import filesystem
from ext.pydaily import filesystem
# from pyslide import pyramid
from ext.pyslide import pyramid
# =======

def get_slide_filenames(slides_dir, ignore_dirs=[]):
	slide_list = filesystem.find_ext_files(slides_dir, ["svs", "SVS"], ignore_dirs)
	slide_filenames = [os.path.splitext(ele)[0].removeprefix(slides_dir + os.sep) for ele in slide_list]
	return slide_filenames


def save_mask_compare(slides_dir, slide_filenames, mask_save_dir=None):
	if mask_save_dir is None: mask_save_dir = os.path.join(os.path.dirname(slides_dir), "visualization/masks")
	slide_num = len(slide_filenames)
	filesystem.overwrite_dir(mask_save_dir)
	for ind in np.arange(slide_num):
		print("processing {}/{}".format(ind+1, slide_num))
		check_slide_mask(slides_dir, slide_filenames, ind, mask_save_dir)


def check_slide_mask(slides_dir, slide_filenames, slide_index, mask_save_dir, tissue_mask_dir, display_level=2):
	""" Load slide segmentation mask.

	"""
	if not os.path.exists(mask_save_dir):
		os.makedirs(mask_save_dir)
	slide_path = os.path.join(slides_dir, slide_filenames[slide_index]+".svs")
	if not os.path.exists(slide_path):
		slide_path = os.path.join(slides_dir, slide_filenames[slide_index]+".SVS")
	wsi_head = pyramid.load_wsi_head(slide_path)
	new_size = (wsi_head.level_dimensions[display_level][1], wsi_head.level_dimensions[display_level][0])
	slide_img = wsi_head.read_region((0, 0), display_level, wsi_head.level_dimensions[display_level])
	slide_img = np.asarray(slide_img)[:,:,:3]

	# load and reize tissue mask
	tissue_mask_path = os.path.join(tissue_mask_dir, slide_filenames[slide_index]+"_tissue.tif")
	tissue_mask_img = io.imread(tissue_mask_path)
	resize_tissue_mask = (transform.resize(tissue_mask_img, new_size, order=0) * 255).astype(np.uint8)
	resize_tissue_mask = np.where(resize_tissue_mask > 0, 255, resize_tissue_mask)
	# load and resize whole mask
	# whole_mask_path = os.path.join(slides_dir, slide_filenames[slide_index]+"_whole.tif")
	# whole_mask_img = io.imread(whole_mask_path) * 255
	# resize_whole_mask = (transform.resize(whole_mask_img, new_size, order=0) * 255).astype(np.uint8)
	# resize_whole_mask = np.where(resize_whole_mask > 0, 255, resize_whole_mask)
	# load and resize viable mask
	viable_mask_path = os.path.join(slides_dir, slide_filenames[slide_index]+"_viable.tif")
	viable_mask_img = io.imread(viable_mask_path) * 255
	resize_viable_mask = (transform.resize(viable_mask_img, new_size, order=0)* 255).astype(np.uint8)
	resize_viable_mask = np.where(resize_viable_mask > 0, 255, resize_viable_mask)
	# show the mask
	fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(24, 7.5))
	ax1.imshow(slide_img)
	ax1.set_title('Slide Image')
	ax2.imshow(resize_tissue_mask, cmap='Greys_r')
	ax2.set_title('Tissue Mask')
	ax3.imshow(resize_viable_mask, cmap='Greys_r')
	ax3.set_title('Viable Tumor Mask')
	plt.tight_layout()
	# plt.show()
	save_path = os.path.join(mask_save_dir, slide_filenames[slide_index]+".png")
	fig.savefig(save_path)
	plt.close(fig)

if __name__ == "__main__":
	slides_dir = "../data/dataset/train"
	slide_filenames = get_slide_filenames(slides_dir)
	mask_save_dir = "../data/visualization/train/masks"
	tissue_mask_dir = "../data/tissue_masks"
	check_slide_mask(slides_dir, slide_filenames, 29, mask_save_dir, tissue_mask_dir)
	#save_mask_compare(slides_dir, slide_filenames, mask_save_dir)
