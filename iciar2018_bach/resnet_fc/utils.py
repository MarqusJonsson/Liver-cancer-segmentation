import datetime
import os
import torch
from torch.utils.data import DataLoader
from dataset import ICIAR2018Dataset

def get_loaders(
	train_dir,
	val_dir,
	batch_size,
	train_transform,
	val_transform,
	num_workers=4,
	pin_memory=True,
):
	train_ds = ICIAR2018Dataset(
		image_dir=train_dir,
		transform=train_transform,
	)

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
		shuffle=True,
	)

	val_ds = ICIAR2018Dataset(
		image_dir=val_dir,
		transform=val_transform,
	)

	val_loader = DataLoader(
		val_ds,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
		shuffle=False,
	)

	return train_loader, val_loader

def get_test_loader(
	test_dir,
	batch_size,
	test_transform,
	num_workers=4,
	pin_memory=True,
):
	test_ds = ICIAR2018Dataset(
		image_dir=test_dir,
		transform=test_transform,
	)

	test_loader = DataLoader(
		test_ds,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
		shuffle=False,
	)

	return test_loader

def calc_dice(predictions, targets, smooth=1e-5):
    # calculate intersection and union of the binary tensors
    intersection = torch.sum(predictions * targets)
    union = torch.sum(predictions) + torch.sum(targets)

    # calculate Dice coefficient
    dice = (2 * intersection + smooth) / (union + smooth)

    return dice

def calc_jaccard(predictions, targets):
	# calculate intersection and union of the binary tensors
	intersection = torch.sum(predictions * targets)
	union = torch.sum(predictions) + torch.sum(targets) - intersection

	# calculate Jaccard score
	jaccard = intersection / union

	return jaccard

def load_checkpoint(checkpoint, model):
		print("=> Loading checkpoint")
		model.load_state_dict(checkpoint["state_dict"])

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint...")
    torch.save(state, filename)


def save_dict_as_csv(filename, dict, delimiter="|"):
    data_to_save = ""
    if not os.path.isfile(filename):
        data_to_save += delimiter.join(key for key in dict.keys()) + "\n"
    data_to_save += delimiter.join(str(value.item()) if torch.is_tensor(value) else str(value) for value in dict.values()) + "\n"
    with open(filename, "a") as f:
        f.write(data_to_save)

def getDatetime():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
