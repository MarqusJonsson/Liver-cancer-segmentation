import datetime
import os
import torch
import torchvision
from dataset import PAIP2019Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
	print("Saving checkpoint...")
	torch.save(state, filename)

def load_checkpoint(checkpoint, model):
	print("=> Loading checkpoint")
	model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
	train_dir,
	train_mask_dir,
	val_dir,
	val_mask_dir,
	batch_size,
	train_transform,
	val_transform,
	num_workers=4,
	pin_memory=True,
):
	train_ds = PAIP2019Dataset(
		image_dir=train_dir,
		mask_dir=train_mask_dir,
		transform=train_transform,
	)

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
		shuffle=True,
	)

	val_ds = PAIP2019Dataset(
		image_dir=val_dir,
		mask_dir=val_mask_dir,
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
	test_mask_dir,
	batch_size,
	test_transform,
	num_workers=4,
	pin_memory=True,
):
	test_ds = PAIP2019Dataset(
		image_dir=test_dir,
		mask_dir=test_mask_dir,
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

# def calc_jaccard_clipped(preds, targets, threshold=0.65):
# 	score = smp.metrics.functional.jaccard(preds, targets)
# 	if score < threshold: return 0
# 	return score

def check_performance(loader, model, loss_fn, result_prefix="", device="cuda"):
	num_correct = 0
	num_pixels = 0
	total_dice_score = 0
	total_jaccard_score = 0
	total_jaccard_score_clipped = 0
	jaccard_score_clip_threshold = 0.65
	total_loss = 0
	model.eval()

	with torch.no_grad():
		loop = tqdm(loader)
		for batch_idx, (x, y) in enumerate(loop):
			x = x.float().to(device)
			y = y.float().to(device)
			predictions = model(x)
			total_loss += loss_fn(predictions, y)
			predictions = torch.sigmoid(predictions)
			#  convert to binary tensors
			predictions = (predictions > 0.5).float()
			accuracy = torch.mean((predictions == y).float())
			total_dice_score += calc_dice(predictions, y)
			jaccard_score = calc_jaccard(predictions, y)
			total_jaccard_score += jaccard_score
			if jaccard_score >= jaccard_score_clip_threshold:
				total_jaccard_score_clipped += jaccard_score

	result = {
		result_prefix + "loss": total_loss/len(loader),
		result_prefix + "dice": total_dice_score/len(loader),
		result_prefix + "acc": accuracy, # num_correct/num_pixels,
		result_prefix + "jaccard": total_jaccard_score/len(loader),
		result_prefix + "jaccard_clipped": total_jaccard_score_clipped/len(loader),
	}

	model.train()
	return result

def save_dict_as_csv(filename, dict, delimiter="|"):
	data_to_save = ""
	if not os.path.isfile(filename):
		data_to_save += delimiter.join(key for key in dict.keys()) + "\n"
	data_to_save += delimiter.join(str(value.item()) if torch.is_tensor(value) else str(value) for value in dict.values()) + "\n"
	with open(filename, "a") as f:
		f.write(data_to_save)


def save_predictions_as_imgs(
	loader, model, folder="saved_images/", device="cuda", max_imgs=5
):
	if not os.path.exists(folder):
		os.makedirs(folder)
	model.eval()
	for idx, (x, y) in enumerate(loader):
		if idx >= max_imgs:
			break
		x = x.float().to(device=device)
		with torch.no_grad():
			preds = torch.sigmoid(model(x))
			preds = (preds > 0.5).float()
		torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
		torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/gt_{idx}.png")
	
	model.train()

def getDatetime():
	return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")