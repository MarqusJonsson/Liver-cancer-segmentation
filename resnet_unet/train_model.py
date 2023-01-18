import os
GPUS =  "5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
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
	getDatetime,
	save_dict_as_csv,
	load_checkpoint,
	save_checkpoint,
	get_loaders,
	save_predictions_as_imgs,
)
import time

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-7
BATCH_SIZE = 16
MAX_NUM_EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "../data/patches/ps_1024_po_0.5_mt_0.8/train/tissue"
TRAIN_MASK_DIR = "../data/patches/ps_1024_po_0.5_mt_0.8/train/viable"
VAL_IMG_DIR = "../data/patches/ps_1024_po_0.5_mt_0.8/val/tissue"
VAL_MASK_DIR = "../data/patches/ps_1024_po_0.5_mt_0.8/val/viable"
ENCODER = "resnet50"
ENCODER_WEIGHT_INITIALIZATION = "imagenet" # None = random weight initialization, "imagenet" = imagenet weight innitialization
BCE_WEIGHT = 0.1
MAX_EPOCHS_WITHOUT_IMPROVEMENT = 5
TRAINING_INFO_FILENAME = "results.csv"
PARAMETER_INFO_FILENAME = "parameters.csv"

def train_fn(loader, model, optimizer, loss_fn, scaler):
	print("Training model...")
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

def check_and_save_performance(train_loader, val_loader, model, loss_fn, epoch, train_time, filename=TRAINING_INFO_FILENAME):
	# check training set performance
	print("Checking training set performance...")
	train_set_performance_start_time = time.time()
	train_set_performance = check_performance(train_loader, model, loss_fn, result_prefix="train_", device=DEVICE)
	check_train_set_performance_time = time.time() - train_set_performance_start_time

	# check validation set perfromance
	print("Checking validation set performance...")
	val_set_performance_start_time = time.time()
	val_set_performance = check_performance(val_loader, model, loss_fn, result_prefix="val_", device=DEVICE)
	check_val_set_perfromance_time = time.time() - val_set_performance_start_time

	currentDatetime = getDatetime()

	training_info = {
		"epoch": epoch,
		**train_set_performance,
		**val_set_performance,
		"train_time": train_time,
		"check_train_set_performance_time": check_train_set_performance_time,
		"check_val_set_performance_time": check_val_set_perfromance_time,
		"datetime": currentDatetime,
	}

	# save training info to csv file
	save_dict_as_csv(filename, training_info)

	return training_info

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
	model = smp.Unet(encoder_name=ENCODER, in_channels=3, classes=1, encoder_weights=ENCODER_WEIGHT_INITIALIZATION)
	model = torch.nn.DataParallel(model)
	model = model.to(device=DEVICE)

	def calc_loss(pred, target, bce_weight=BCE_WEIGHT):
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
		load_checkpoint(torch.load("my_checkpoint.pt"), model)
	training_info = check_and_save_performance(train_loader, val_loader, model, loss_fn, epoch=0, train_time=0)

	scaler = torch.cuda.amp.GradScaler()

	epochs_since_last_improvement = 0
	best_val_loss = training_info["val_loss"]
	# Save hyperparameters
	save_dict_as_csv(
		PARAMETER_INFO_FILENAME, {
			"GPUS" : GPUS,
			"DEVICE" : DEVICE,
			"LEARNING_RATE" : LEARNING_RATE,
			"BATCH_SIZE" : BATCH_SIZE,
			"NUM_EPOCHS" : MAX_NUM_EPOCHS,
			"NUM_WORKERS" : NUM_WORKERS,
			"PIN_MEMORY" : PIN_MEMORY,
			"LOAD_MODEL" : LOAD_MODEL,
			"TRAIN_IMG_DIR" : TRAIN_IMG_DIR,
			"TRAIN_MASK_DIR" : TRAIN_MASK_DIR,
			"VAL_IMG_DIR" : VAL_IMG_DIR,
			"VAL_MASK_DIR" : VAL_MASK_DIR,
			"ENCODER" : ENCODER,
			"ENCODER_WEIGHT_INITIALIZATION" : ENCODER_WEIGHT_INITIALIZATION,
			"BCE_WEIGHT" : BCE_WEIGHT,
			"TRAINING_INFO_FILENAME" : TRAINING_INFO_FILENAME,
			"PARAMETER_INFO_FILENAME" : PARAMETER_INFO_FILENAME,
			"MAX_EPOCHS_WITHOUT_IMPROVEMENT" : MAX_EPOCHS_WITHOUT_IMPROVEMENT,
		}
	)
	model.train()
	for epoch in range(MAX_NUM_EPOCHS):
		# Train model
		train_start_time = time.time()
		train_fn(train_loader, model, optimizer, loss_fn, scaler)
		train_time = time.time() - train_start_time

		# Check and save model performance
		training_info = check_and_save_performance(train_loader, val_loader, model, loss_fn, epoch + 1, train_time)

		current_val_loss = training_info['val_loss']
		# Save model checkpoint
		checkpoint = {
			"state_dict": model.state_dict(),
			"optimizer": optimizer.state_dict(),
		}
		save_checkpoint(checkpoint, f"e_{epoch + 1}_l_{current_val_loss:.4f}_d_{training_info['datetime']}.pt")

		if current_val_loss < best_val_loss:
			best_val_loss = current_val_loss
			epochs_since_last_improvement = 0
		else:
			epochs_since_last_improvement += 1
		print(f"Epochs without improvment: {epochs_since_last_improvement}/{MAX_EPOCHS_WITHOUT_IMPROVEMENT}")		
		# print some examples to a folder
		#save_predictions_as_imgs(val_loader, model, folder="pred_samples/", device=DEVICE)
		if epochs_since_last_improvement >= MAX_EPOCHS_WITHOUT_IMPROVEMENT:
			break

if __name__ == "__main__":
	# torch.cuda.set_device(0)
	# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	main()