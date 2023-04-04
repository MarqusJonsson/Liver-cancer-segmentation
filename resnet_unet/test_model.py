import os
GPUS =  "1,3,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,6,7"
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from utils import (
	calc_dice,
	calc_jaccard,
	check_performance,
	getDatetime,
	load_checkpoint,
	save_checkpoint,
	get_test_loader,
	save_dict_as_csv,
	save_predictions_as_imgs,
)
import time;
import augmentations as aug
# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
TEST_IMG_DIR = "../../patches/ps_1024_po_0.5_mt_0.8/test/tissue"
TEST_MASK_DIR = "../../patches/ps_1024_po_0.5_mt_0.8/test/viable"
ENCODER = "resnet50"
BCE_WEIGHT = 0.1
TEST_INFO_FILENAME = "test_results.csv"
MODEL_LOAD_PATH = "exp/vicreg_viable/e_8_l_0.3803_d_20230329T223949Z.pt"

def main():
	test_transforms = aug.TrainTransform()

	model = smp.Unet(encoder_name=ENCODER, in_channels=3, classes=1).to(device=DEVICE)
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

	test_loader = get_test_loader(
		TEST_IMG_DIR,
		TEST_MASK_DIR,
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