import openslide

__all__ = [	"load_wsi_head",
			]

def load_wsi_head(wsi_img_path):
	""" Load the header meta data of whole slide pyramidal image.
	Parameters
	-------
	wsi_img_path: str
		The path to whole slide image
	Returns
	-------
	wsi_head: slide metadata
		Meta information of whole slide image
	"""
	wsi_head = openslide.OpenSlide(wsi_img_path)
	return wsi_head