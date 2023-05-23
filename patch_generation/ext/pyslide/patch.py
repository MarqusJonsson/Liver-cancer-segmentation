from skimage import color

def patch_bk_ratio(img, bk_thresh=0.80):
	""" Calculate the ratio of background in the image
	Parameters
	-------
	img: np.array
		patch image
	bk_thresh: float
		background threshold value
	Returns
	-------
	bk_ratio: float
		the ratio of background in a patch
	"""

	g_img = color.rgb2gray(img)
	bk_num = (g_img > bk_thresh).sum()
	pixel_num = g_img.shape[0] * g_img.shape[1]
	bk_ratio = bk_num * 1.0 / pixel_num

	return bk_ratio