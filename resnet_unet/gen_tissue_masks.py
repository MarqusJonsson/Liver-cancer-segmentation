import numpy as np
from threading import Thread
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage import label
from tqdm import trange
from skimage.measure import regionprops
import os, warnings, pyvips, tifffile
warnings.filterwarnings('ignore', category=UserWarning)

w_size = 256
stride = 256

total = w_size**2
min_px = total // 10
rm_min_size = 10
rm_min_hole_size = 10
pad_px = w_size * 2
max_rgb = [235, 210, 235]
threshold_vote = 0

def extract_coord(coord_str):
	y_start = coord_str.split('x')[0]
	x_start = coord_str.split('x')[1]
	return int(y_start), int(x_start)

def find_minmax(img):
	rps = regionprops(img)
	assert(len(rps) == 1)
	rp = rps[0]
	y_min, x_min, y_max, x_max = rp.bbox
	y_min = max(y_min - pad_px, 0)
	y_max = min(y_max + pad_px, img.shape[0])
	x_min = max(x_min - pad_px, 0)
	x_max = min(x_max + pad_px, img.shape[1])
	return (y_min, x_min, y_max, x_max)

def unit_threshold_and_amend(single_ch_img, threshold, ret):
	temp = single_ch_img < threshold
	temp = remove_small_holes(label(temp)[0], area_threshold=rm_min_hole_size)>0
	temp = remove_small_objects(label(temp)[0], min_size=rm_min_size)>0
	ret += temp

def threshold_and_amend(img, ret):
	board = np.zeros(img.shape[:-1], dtype=np.uint8)
	threads = []
	for i in range(3):
		t = Thread(target=unit_threshold_and_amend, args=(img[:, :, i], max_rgb[i], board))
		threads.append(t)
		t.start()
	for t in threads:
		t.join()
	ret += (board>threshold_vote).astype(np.uint8)

def find_foreground(img):
	threshold_tissue = np.zeros(img.shape[:-1], dtype=np.uint8)
	threads = []
	t = Thread(target=threshold_and_amend, args=(img, threshold_tissue))
	threads.append(t)
	t.start()
	for t in threads:
		t.join()
	tissue = (threshold_tissue > threshold_vote).astype(np.uint8)
	return tissue

# https://github.com/libvips/pyvips/blob/master/examples/pil-numpy-pyvips.py
# map vips formats to np dtypes
format_to_dtype = {
	'uchar': np.uint8,
	'char': np.int8,
	'ushort': np.uint16,
	'short': np.int16,
	'uint': np.uint32,
	'int': np.int32,
	'float': np.float32,
	'double': np.float64,
	'complex': np.complex64,
	'dpcomplex': np.complex128,
}

# vips image to numpy array
def vips2numpy(vi):
	return np.ndarray(buffer=vi.write_to_memory(),
		dtype=format_to_dtype[vi.format],
		shape=[vi.height, vi.width, vi.bands])


if __name__ == '__main__':

	slides_dir = "../data/dataset/train"
	slide_list = [os.path.splitext(ele)[0] for ele in os.listdir(slides_dir) if "svs" in ele]
	slide_list.sort()
	tissue_masks_dir = "../data/tissue_masks"

	if not os.path.isdir(tissue_masks_dir):
		os.makedirs(tissue_masks_dir)

	for idx in trange(0, len(slide_list)):
		slide_path = os.path.join(slides_dir, slide_list[idx] + ".svs")
		tissue_mask_path = os.path.join(tissue_masks_dir, slide_list[idx] + "_tissue.tif")

		if not os.path.isfile(slide_path):
			print('wsi not found:', slide_path)
			exit()

		# print(f"loading {slide_path}...")
		img = vips2numpy(pyvips.Image.new_from_file(slide_path))[:, :, :3]
		# print("generating tissue mask...")
		tissue = (find_foreground(img) * 255).astype(np.uint8)
		# print("saving tissue mask img...")
		tifffile.imwrite(tissue_mask_path, tissue, compression=tifffile.tifffile.TIFF.COMPRESSION.ADOBE_DEFLATE)