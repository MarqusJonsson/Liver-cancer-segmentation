import os
GPUS =  "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
import torch
from torch import nn
from torchvision.models import resnet50
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
	calc_dice,
	calc_jaccard,
	getDatetime,
	load_checkpoint,
	get_test_loader,
	save_dict_as_csv,
)
import time
import augmentations as aug
# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
TEST_IMG_DIR = "../../data/patches/iciar2018_ps_512_po_0.8/test"
ENCODER = "resnet50"
BCE_WEIGHT = 0.1
TEST_INFO_FILENAME = "test_results.csv"
MODEL_LOAD_PATH = "exp/vicreg_aug_e10_unfreeze/e_23_l_0.4030_d_20231220T191317Z.pt"
#"exp/vicreg_50po/e_52_l_0.6224_d_20231216T073717Z.pt"
#"exp/vicreg_conformal/e_184_l_0.4960_d_20231215T204616Z.pt"
#"exp/vicreg_aug_e10/e_130_l_0.4774_d_20231212T115732Z.pt"
#"exp/random_3/e_242_l_0.5741_d_20231214T073207Z.pt"
#"exp/imagenet_1k_v2/e_145_l_0.5128_d_20231215T022436Z.pt"

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

			predictions = torch.sigmoid(predictions)

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
	test_transforms = aug.TrainTransform()

	model = resnet50(weights=None)

	# Define the number of output neurons
	num_classes = 4
	# Create the FC layer
	fc_layer = nn.Linear(model.fc.in_features, num_classes)
	# Add the FC layer to the end of the model
	model.fc = fc_layer

	model = torch.nn.DataParallel(model)
	model = model.to(device=DEVICE)

	def calc_loss(pred, target, bce_weight=BCE_WEIGHT):
		cce = F.cross_entropy(pred, target)
		# bce = F.binary_cross_entropy_with_logits(pred, target)
		# dice_score = calc_dice(pred, target)

		loss = cce#bce * bce_weight + (1.0 - dice_score) * (1.0 - bce_weight)
		return loss

	loss_fn = calc_loss

	test_loader = get_test_loader(
		TEST_IMG_DIR,
		BATCH_SIZE,
		test_transforms,
		NUM_WORKERS,
		PIN_MEMORY,
	)

	load_checkpoint(torch.load(MODEL_LOAD_PATH), model)
	# check test set perfromance
	print("Checking test set performance...")
	test_set_performance_start_time = time.time()
	test_set_performance = check_performance(test_loader, model, loss_fn, result_prefix="test_", device=DEVICE)
	check_test_set_perfromance_time = time.time() - test_set_performance_start_time
	currentDatetime = getDatetime()

	test_info = {
		**test_set_performance,
		"check_test_set_perfromance_time": check_test_set_perfromance_time,
		"datetime": currentDatetime,
		"model_path": MODEL_LOAD_PATH,
	}

	# save test info to csv file
	save_dict_as_csv(TEST_INFO_FILENAME, test_info)

	# print some examples to a folder
	# save_predictions_as_imgs(test_loader, model, folder="pred_samples/", device=DEVICE)

if __name__ == "__main__":
	# torch.cuda.set_device(7)
	main()
