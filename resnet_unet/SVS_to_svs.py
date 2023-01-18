import os

if __name__ == "__main__":
	slides_dir = "../data/dataset/train"
	slide_list = [slide_path for slide_path in os.listdir(slides_dir) if "SVS" in slide_path]
	n_slides = len(slide_list)
	print(f"Renaming {n_slides} files...")
	for slide_path in slide_list:
		prev = os.path.join(slides_dir, slide_path)
		new = os.path.join(slides_dir, slide_path[:-4] + ".svs")
		os.rename(prev, new)
		print(f"\"{prev}\" -> \"{os.path.join(new)}\"")