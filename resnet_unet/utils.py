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

def calc_dice(preds, targets):
	# smooth = 1
	# pflat = preds.view(-1)
	# tflat = targets.view(-1)
	# dice = ((2. * (pflat * tflat).sum() + smooth) / (pflat.sum() + tflat.sum() + smooth))
	return smp.utils.metrics.F.f_score(preds, targets)

def calc_jaccard(preds, targets):
	# intersection = (preds * targets).sum()
	# return intersection / ((preds + targets).sum() - intersection + 1e-8)
	return smp.utils.metrics.F.jaccard(preds, targets)

def calc_jaccard_clipped(preds, targets, threshold=0.65):
	score = smp.utils.metrics.F.jaccard(preds, targets)
	if score < threshold: return 0
	return score

def check_performance(loader, model, loss_fn, result_prefix="", device="cuda"):
	num_correct = 0
	num_pixels = 0
	dice_score = 0
	jaccard_score = 0
	jaccard_score_clipped = 0
	jaccard_score_clip_threshold = 0.65
	loss = 0
	model.eval()

	with torch.no_grad():
		loop = tqdm(loader)
		for batch_idx, (x, y) in enumerate(loop):
			x = x.float().to(device)
			y = y.float().to(device).unsqueeze(1)
			preds = model(x)
			loss += loss_fn(preds, y)
			preds = torch.sigmoid(preds)
			preds = (preds > 0.5).float()
			num_correct += (preds == y).sum()
			num_pixels += torch.numel(preds)
			# y_cpu = y.cpu().numpy().flatten()
			# preds_cpu = preds.cpu().numpy().flatten()
			# f1_s += f1_score(y.flatten(), preds.flatten())
			# jaccard_scr2 += jaccard_score(y.cpu().flatten(), preds.cpu().flatten())
			dice_score += calc_dice(preds, y)
			j_s = calc_jaccard(preds, y)
			jaccard_score += j_s
			if j_s >= jaccard_score_clip_threshold:
				jaccard_score_clipped += j_s

	result = {
		result_prefix + "loss": loss/len(loader),
		result_prefix + "dice": dice_score/len(loader),
		result_prefix + "acc": num_correct/num_pixels,
		result_prefix + "jaccard": jaccard_score/len(loader),
		result_prefix + "jaccard_clipped": jaccard_score_clipped/len(loader),
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