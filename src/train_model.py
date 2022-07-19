import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
import segmentation_models_pytorch as smp
from utils import (
	calc_dice,
	calc_jaccard,
	check_performance,
	load_checkpoint,
	save_checkpoint,
	get_loaders,
	save_predictions_as_imgs,
)

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "../data/patches/ps_1024_po_0.5_mt_0.8/train/tissue"
TRAIN_MASK_DIR = "../data/patches/ps_1024_po_0.5_mt_0.8/train/whole"
VAL_IMG_DIR = "../data/patches/ps_1024_po_0.5_mt_0.8/val/tissue"
VAL_MASK_DIR = "../data/patches/ps_1024_po_0.5_mt_0.8/val/whole"

def train_fn(loader, model, optimizer, loss_fn, scaler):
	loop = tqdm(loader)

	for batch_idx, (data, targets) in enumerate(loop):
		data = data.float().to(device=DEVICE)
		targets = targets.float().unsqueeze(1).to(device=DEVICE)

		# forward
		with torch.cuda.amp.autocast():
			predictions = model(data)
			loss = loss_fn(predictions, targets)

		# backward
		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		# update tqdm loop
		loop.set_postfix(loss=loss.item())

def main():
	train_transforms = A.Compose([
		# A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
		# A.Rotate(limit=35, p=1.0),
		# A.HorizontalFlip(p=0.5),
		# A.VerticalFlip(p=0.1),
		# ToTensor doesn't divide by 255 like PyTorch,
		# it's done inside normalize function
		# A.Normalize(
		# 	mean=[0.0, 0.0, 0.0],
		# 	std=[1.0, 1.0, 1.0],
		# 	max_pixel_value=255.0,
		# ),
		ToTensorV2(),
	])
	val_transforms = A.Compose([
		# A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
		# ToTensor doesn't divide by 255 like PyTorch,
		# it's done inside normalize function
		# A.Normalize(
		# 	mean=[0.0, 0.0, 0.0],
		# 	std=[1.0, 1.0, 1.0],
		# 	max_pixel_value=255.0,
		# ),
		ToTensorV2(),
	])

	# model = UNet(in_channels=3, out_channels=1).to(device=DEVICE)
	model = smp.Unet(encoder_name="resnet50", in_channels=3, classes=1, encoder_weights="imagenet")
	model = torch.nn.DataParallel(model)
	model = model.to(device=DEVICE)

	def calc_loss(pred, target, bce_weight=0.1):
		bce = F.binary_cross_entropy_with_logits(pred, target)

		pred = torch.sigmoid(pred)
		dice = calc_dice(pred, target)
		# jaccard = calc_jaccard(pred, target)
		loss = bce * bce_weight + (1.0 - dice) * (1.0 - bce_weight)

		return loss

	loss_fn = calc_loss
	# loss_fn = nn.BCEWithLogitsLoss() # Logits since we dont sigmoid output of model
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	train_loader, val_loader = get_loaders(
		TRAIN_IMG_DIR,
		TRAIN_MASK_DIR,
		VAL_IMG_DIR,
		VAL_MASK_DIR,
		BATCH_SIZE,
		train_transforms,
		val_transforms,
		NUM_WORKERS,
		PIN_MEMORY,
	)

	if LOAD_MODEL:
		load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
	check_performance(val_loader, model, device=DEVICE)

	scaler = torch.cuda.amp.GradScaler()

	best_jaccard = 0
	for epoch in range(NUM_EPOCHS):
		train_fn(train_loader, model, optimizer, loss_fn, scaler)

		# save model
		checkpoint = {
			"state_dict": model.state_dict(),
			"optimizer": optimizer.state_dict(),
		}

		# check accuracy
		result_metrics = check_performance(val_loader, model, device=DEVICE)

		if result_metrics["jaccard"] > best_jaccard:
			best_jaccard = result_metrics["jaccard"]
			save_checkpoint(checkpoint, f"e_{epoch}_j_{best_jaccard:.4f}_d_{result_metrics['dice']:.4f}_a_{result_metrics['acc']:.4f}.pth.tar")

		# print some examples to a folder
		save_predictions_as_imgs(val_loader, model, folder="pred_samples/", device=DEVICE)

if __name__ == "__main__":
	# torch.cuda.set_device(0)
	# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	main()