import os
import shutil

__all__ = ['find_ext_files',
           'overwrite_dir'
		   ]

def find_ext_files(dir_name, exts, ignore_dirs=[]):
	""" Find all files with given extension in a given directory
	Parameters
	-------
	dir_name: str
		given directory to locate files
	exts: list
		file extensions
	Returns
	-------
	file_list: list
		files with specified extesion in given directory
	"""
	if not os.path.isdir(dir_name):
		raise AssertionError("{} is not a valid directory".format(dir_name))
	file_list = []
	for root, _, files in os.walk(dir_name, followlinks=True):
		if root.split(os.sep)[-1] in ignore_dirs: continue
		for cur_file in files:
			for ext in exts:
				if cur_file.endswith(ext):
					file_list.append(os.path.join(root, cur_file))
					break
	return file_list

def overwrite_dir(dir_name):
	""" Overwrite directory if exist, create new directory
	"""
	if os.path.exists(dir_name):
		shutil.rmtree(dir_name)
	os.makedirs(dir_name)