import os
import time
from typing import Dict, List, Set

class PatchData:
	# Class for storing information about a patch
	def __init__(self, full_filename: str, parent_name: str, slide_index: int, x: int, y: int, file_extension: str):
		"""
		Initialize the fields of the PatchData class
		"""
		self.full_filename: str = full_filename
		self.parent_name: str = parent_name # name of original image which the patch was created from
		self.slide_index: int = slide_index
		self.x: int = x
		self.y: int = y
		self.file_extension: str = file_extension

def create_patch_neighbour_dict(filenames: List[str], patch_size: int = 1024, patch_overlap: int = 0.5) -> Dict[str, List[str]]:
	"""
	Given a list of filenames, returns a dictionary whose keys are filenames
	and values are lists of filenames of neighbouring patches.
	"""
	patch_neighbour_dict = {}
	filenames_set = set(filenames)
	for filename in filenames_set:
		patch_neighbour_dict[filename] = find_neighbours(filename, filenames_set, patch_size, patch_overlap)
	return patch_neighbour_dict

def find_neighbours(filename: str, filenames_set: Set[str], patch_size: int, patch_overlap: int) -> List[str]:
	"""
	Given a filename and a set of filenames, returns a list of filenames that
	are considered to be neighbours of the input filename.
	"""
	patch_offset = int(patch_size * (1.0 - patch_overlap))
	neighbours = []
	patch_data = extract_patch_data(filename)
	for x_offset in range(-patch_offset, patch_offset + 1, patch_offset):
		for y_offset in range(-patch_offset, patch_offset + 1, patch_offset):
			# Skip the case where offset is zero
			if x_offset == 0 and y_offset == 0:
				continue
			# Skip diagonal neighbours
			if x_offset == y_offset or x_offset == -y_offset:
				continue
			# Construct a potential neighbouring filename
			neighbour = "_".join([
					patch_data.parent_name,
					str(patch_data.slide_index),
					str(patch_data.x + x_offset),
					str(patch_data.y + y_offset)
				]) + "." + patch_data.file_extension
			if neighbour in filenames_set:
				neighbours.append(neighbour)
	return neighbours

def extract_patch_data(filename: str) -> PatchData:
	"""
	Given a filename, returns an instance of the PatchData class,
	by extracting the relevant information from the filename.
	"""
	name, extension = filename.split(".")
	split_name = name.split("_")
	return PatchData(filename, "_".join(split_name[0:3]), int(split_name[3]), int(split_name[4]), int(split_name[5]), extension)


if __name__ == "__main__":
	patch_directory = "../data/patches/ps_1024_po_0.5_mt_0.8/train/tissue"
	start = time.time()
	patches = os.listdir(patch_directory)
	end = time.time()
	print("Time to create patch list:", end - start)

	start = time.time()
	patches = create_patch_neighbour_dict(patches)
	end = time.time()
	print("Time to create patch neighbour dict: ", end - start)
	for patch in patches:
		neighbours = patches.get(patch)
		#neighbours = getNeighbouringPatches(patch, patches, 1024, 0.5)
		print(patch, " has ", len(neighbours), " neighbours: ", neighbours, sep="")
