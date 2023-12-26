import os
GPUS = "1,4"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
import augmentations as aug
import time
from utils import (
    calc_dice,
    calc_jaccard,
    get_loaders,
    getDatetime,
    load_checkpoint,
    save_checkpoint,
    save_dict_as_csv
)

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-8
BATCH_SIZE = 16
MAX_NUM_EPOCHS = 200
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "../../data/patches/iciar2018_ps_512_po_0.8/train"
VAL_IMG_DIR = "../../data/patches/iciar2018_ps_512_po_0.8/val"
ENCODER = "resnet50"
ENCODER_WEIGHT_INITIALIZATION = None # None = random weight initialization, "imagenet" = imagenet weight innitialization
BCE_WEIGHT = 0.1
LABEL_SMOOTHING = 0.2
MAX_EPOCHS_WITHOUT_IMPROVEMENT = 20
TRAINING_INFO_FILENAME = "results.csv"
PARAMETER_INFO_FILENAME = "parameters.csv"
LOAD_MODEL_PATH = "../../resnet_unet/models/vicreg_aug_80_patch_overlap_resnet50_epoch_10.pth"

SAVE_DIRECTORY = "exp/vicreg_aug_e10_unfreeze_loss_cce_lower_lr_label_smooth_0.2"

def train_fn(loader, model, optimizer, loss_fn, scaler):
	print("Training model...")
	loop = tqdm(loader)

	for batch_idx, (data, targets, _) in enumerate(loop):
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
	save_dict_as_csv(SAVE_DIRECTORY + "/" + filename, training_info)

	return training_info

def check_performance(loader, model, loss_fn, result_prefix="", device="cuda"):
	total_loss = 0
	total_accuracy = 0
	model.eval()

	result_dict = {}

	with torch.no_grad():
		loop = tqdm(loader)
		for batch_idx, (x, y, patch_parent_image_name) in enumerate(loop):
			x = x.float().to(device)
			y = y.float().to(device)
			predictions = model(x)
			total_loss += loss_fn(predictions, y)

			predictions = torch.softmax(predictions, dim=-1)

			# Find the index of the maximum value along the second dimension (axis=1)
			max_values, max_indices = torch.max(predictions, dim=1)
			# Create a new tensor with the same shape as predictions, filled with zeros
			one_hot_predictions = torch.zeros_like(predictions)
			# Use fancy indexing to set the maximum value in each row to 1
			one_hot_predictions[torch.arange(predictions.size(0)), max_indices] = 1

			# Check for complete vector equality
			matching_vectors = (one_hot_predictions == y).all(dim=1)
			# Count the number of matching vectors
			num_matching_vectors = matching_vectors.sum().item()

			total_accuracy += num_matching_vectors / len(one_hot_predictions)


			for sample_index, sample in enumerate(x):
				image_name = patch_parent_image_name[sample_index].split(".")[0]
				if image_name in result_dict:
					result_dict[image_name]["predictions"] = torch.cat((result_dict[image_name]["predictions"], predictions[sample_index].unsqueeze(0)), dim=0)
					result_dict[image_name]["one_hot_predictions"] = torch.cat((result_dict[image_name]["one_hot_predictions"], one_hot_predictions[sample_index].unsqueeze(0)), dim=0)
				else:
					result_dict[image_name] = {"predictions": predictions[sample_index].unsqueeze(0), "one_hot_predictions": one_hot_predictions[sample_index].unsqueeze(0), "label": y[sample_index].unsqueeze(0)}
	
	num_images = 0
	num_correct_one_hot_predictions = 0
	num_correct_predictions = 0
	for image_results_key in result_dict:
		image_results = result_dict[image_results_key]
		total_predictions = torch.sum(image_results["predictions"], dim=0)
		total_one_hot_predictions = torch.sum(image_results["one_hot_predictions"], dim=0)
		total_label = torch.sum(image_results["label"], dim=0)
		num_images += 1
		if total_one_hot_predictions.argmax() == total_label.argmax():
			num_correct_one_hot_predictions += 1
		if total_predictions.argmax() == total_label.argmax():
			num_correct_predictions += 1

	result = {
		result_prefix + "loss": total_loss / len(loader),
		result_prefix + "patch_acc": total_accuracy / len(loader),
		result_prefix + "image_acc": num_correct_predictions / num_images,
		result_prefix + "image_acc_one_hot": num_correct_one_hot_predictions / num_images,
	}

	model.train()
	return result

def main():
	if not os.path.exists(SAVE_DIRECTORY):
		os.makedirs(SAVE_DIRECTORY)

	model = resnet50(weights=None)

	# Freeze parameters
	# for param in model.parameters():
	# 	param.requires_grad = False

	# Define the number of output neurons
	num_classes = 4
	# Create the FC layer
	fc_layer = nn.Linear(model.fc.in_features, num_classes)
	# Add the FC layer to the end of the model
	model.fc = fc_layer

	model = torch.nn.DataParallel(model)
	model = model.to(device=DEVICE)

	def calc_loss(pred, target, bce_weight=BCE_WEIGHT):
		cce = F.cross_entropy(pred, target, label_smoothing=LABEL_SMOOTHING)
		# bce = F.binary_cross_entropy_with_logits(pred, target)
		# dice_score = calc_dice(pred, target)

		loss = cce#bce * bce_weight + (1.0 - dice_score) * (1.0 - bce_weight)
		return loss

	loss_fn = calc_loss
	
	if LOAD_MODEL:
		# Load vicreg encoder
		# model.module.encoder.load_state_dict(vicreg_state_dict)

		vicreg_state_dict = torch.load(LOAD_MODEL_PATH)
		missing_keys, unexpected_keys = model.module.load_state_dict(vicreg_state_dict, strict=False)

		print("MISSING KEYS:", missing_keys)
		print("UNEXPECTED KEYS:", unexpected_keys)
		# Load saved model
		# load_checkpoint(torch.load(LOAD_MODEL_PATH), model)

	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	train_transforms = aug.TrainTransform()
	val_transforms = aug.TrainTransform()
	train_loader, val_loader = get_loaders(
			TRAIN_IMG_DIR,
			VAL_IMG_DIR,
			BATCH_SIZE,
			train_transforms,
			val_transforms,
			NUM_WORKERS,
			PIN_MEMORY,
	)

	scaler = torch.cuda.amp.GradScaler()

	epochs_since_last_improvement = 0
	best_val_loss = 9999

	# Save hyperparameters
	save_dict_as_csv(
		SAVE_DIRECTORY + "/" + PARAMETER_INFO_FILENAME, {
			"GPUS" : GPUS,
			"DEVICE" : DEVICE,
			"LEARNING_RATE" : LEARNING_RATE,
			"BATCH_SIZE" : BATCH_SIZE,
			"NUM_EPOCHS" : MAX_NUM_EPOCHS,
			"NUM_WORKERS" : NUM_WORKERS,
			"PIN_MEMORY" : PIN_MEMORY,
			"LOAD_MODEL" : LOAD_MODEL,
			"TRAIN_IMG_DIR" : TRAIN_IMG_DIR,
			"VAL_IMG_DIR" : VAL_IMG_DIR,
			"ENCODER" : ENCODER,
			"ENCODER_WEIGHT_INITIALIZATION" : ENCODER_WEIGHT_INITIALIZATION,
			"BCE_WEIGHT" : BCE_WEIGHT,
			"LABEL_SMOOTHING" : LABEL_SMOOTHING,
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

		current_val_loss = training_info["val_loss"]
		# Save model checkpoint
		checkpoint = {
			"state_dict": model.state_dict(),
			"optimizer": optimizer.state_dict(),
		}
		save_checkpoint(checkpoint, SAVE_DIRECTORY + "/" + f"e_{epoch + 1}_l_{current_val_loss:.4f}_d_{training_info['datetime']}.pt")
		
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
