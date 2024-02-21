import os
GPUS = "3,4"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
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
import augmentations as aug
# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-7
BATCH_SIZE = 16
MAX_NUM_EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "../data/patches/ps_1024_po_0.8_mt_0.8_tr80_v10_te10/train/tissue"
TRAIN_MASK_DIR = "../data/patches/ps_1024_po_0.8_mt_0.8_tr80_v10_te10/train/viable"
VAL_IMG_DIR = "../data/patches/ps_1024_po_0.8_mt_0.8_tr80_v10_te10/val/tissue"
VAL_MASK_DIR = "../data/patches/ps_1024_po_0.8_mt_0.8_tr80_v10_te10/val/viable"
ENCODER = "resnet50"
ENCODER_WEIGHT_INITIALIZATION = None # None = random weight initialization, "imagenet" = imagenet weight innitialization
BCE_WEIGHT = 0.1
MAX_EPOCHS_WITHOUT_IMPROVEMENT = 5
TRAINING_INFO_FILENAME = "results.csv"
PARAMETER_INFO_FILENAME = "parameters.csv"
LOAD_MODEL_PATH = "./e_1_l_0.3065_d_20240129T110141Z.pt"#"../vicreg/exp/bs512_is224_20240124/resnet50_epoch_30.pth"#"./models/vicreg_aug_80_patch_overlap_resnet50_epoch_10.pth" # "e_2_l_0.2336_d_20230516T073406Z.pt"

def train_fn(loader, model, optimizer, loss_fn, scaler):
	print("Training model...")
	loop = tqdm(loader)

	for batch_idx, (data, targets) in enumerate(loop):
		data = data.float().to(device=DEVICE)
		targets = targets.float().to(device=DEVICE)

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
	# train_set_performance = check_performance(train_loader, model, loss_fn, result_prefix="train_", device=DEVICE)
	check_train_set_performance_time = time.time() - train_set_performance_start_time

	# check validation set perfromance
	print("Checking validation set performance...")
	val_set_performance_start_time = time.time()
	val_set_performance = check_performance(val_loader, model, loss_fn, result_prefix="val_", device=DEVICE)
	check_val_set_perfromance_time = time.time() - val_set_performance_start_time

	currentDatetime = getDatetime()

	training_info = {
		"epoch": epoch,
		# **train_set_performance,
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
	train_transforms = aug.TrainTransform()
	val_transforms = aug.TrainTransform()

	model = smp.Unet(encoder_name=ENCODER, in_channels=3, classes=1, encoder_weights=ENCODER_WEIGHT_INITIALIZATION)
	model = torch.nn.DataParallel(model)
	model = model.to(device=DEVICE)

	def calc_loss(pred, target, bce_weight=BCE_WEIGHT):
		bce = F.binary_cross_entropy_with_logits(pred, target)

		pred = torch.sigmoid(pred)
		dice_score = calc_dice(pred, target)
		# jaccard = calc_jaccard(pred, target)
		loss = bce * bce_weight + (1.0 - dice_score) * (1.0 - bce_weight)

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
		# Load vicreg encoder
		# vicreg_state_dict = torch.load(LOAD_MODEL_PATH)
		# # model.module.encoder.load_state_dict(vicreg_state_dict)
		# # missing_keys, unexpected_keys = 
		# model.module.encoder.load_state_dict(vicreg_state_dict, strict=True)
		# print("MISSING KEYS:", missing_keys)
		# print("UNEXPECTED KEYS:", unexpected_keys)
		# Load saved model
		load_checkpoint(torch.load(LOAD_MODEL_PATH), model)


		# total_sum = 0
		# for key, value in vicreg_state_dict.items():
		# 	total_sum += torch.sum(value)

		# print("The sum of all state dictionary parameters is:", total_sum.item())

		# state_dict = model.state_dict()

		# total_sum = 0
		# for key, value in state_dict.items():
		# 	total_sum += torch.sum(value)

		# print("The sum of all state dictionary parameters is:", total_sum.item())
 
		# model.module.encoder.load_state_dict(vicreg_state_dict)

		# state_dict = model.state_dict()

		# total_sum = 0
		# for key, value in state_dict.items():
		# 	total_sum += torch.sum(value)

		# print("The sum of all state dictionary parameters is:", total_sum.item())

		# load_checkpoint(torch.load(LOAD_MODEL_PATH), model)
		# print(state_dict.items())
		# print(model.module)

		# exit()
	# training_info = check_and_save_performance(train_loader, val_loader, model, loss_fn, epoch=0, train_time=0)

	scaler = torch.cuda.amp.GradScaler()

	epochs_since_last_improvement = 0
	# best_val_loss = training_info["val_loss"]
	best_val_loss = 99999
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
			"LOAD_MODEL_PATH" : LOAD_MODEL_PATH
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
	main()