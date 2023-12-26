import os
import matplotlib.pyplot as plt
from tqdm import trange
from skimage import io

def save_fig(fig_name, patch_imgs, titles, save_dir):
	fig, axes = plt.subplots(nrows=1, ncols=len(patch_imgs), figsize=(16, 4))
	for idx, patch_img in enumerate(patch_imgs):
		axes[idx].imshow(patch_img)
		axes[idx].set_title(titles[idx])
	plt.tight_layout()
	# plt.show(block=False)
	# plt.pause(3)
	fig_name = str(fig_name)
	fig_name = fig_name.removesuffix(".jpg")
	if not fig_name.endswith(".png"):
		fig_name += ".png"
	plt.savefig(os.path.join(save_dir, fig_name))
	plt.close(fig)

def view_patches(patches_path, save_dir, max_amount=99999):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	patch_types = os.listdir(patches_path)
	patch_paths = {}
	for patch_type in patch_types:
		patch_paths[patch_type] = os.listdir(os.path.join(patches_path, patch_type))
	for idx in trange(min(len(patch_paths[patch_types[0]]), max_amount)):
		if idx >= max_amount:
			break
		patch_imgs = []
		for patch_type in patch_paths:
			patch_imgs.append(io.imread(os.path.join(patches_path, patch_type, patch_paths[patch_type][idx])))
		# patch_mask_img *= 255 # Not needed due to how matplotlib colors the mask image
		save_fig(patch_paths[patch_types[0]][idx], patch_imgs, patch_types, save_dir)

if __name__ == "__main__":
	view_patches("../data/patches/ps_1024_po_0.8_mt_0.8/train", "view_patches", 10)