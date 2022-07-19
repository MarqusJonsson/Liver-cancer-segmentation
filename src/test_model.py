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
	get_test_loader,
	save_predictions_as_imgs,
)

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
TEST_IMG_DIR = "../data/patches/ps_1024_po_0.5_mt_0.8/test/tissue"
TEST_MASK_DIR = "../data/patches/ps_1024_po_0.5_mt_0.8/test/whole"

def main():
	test_transforms = A.Compose([
		ToTensorV2(),
	])

	model = smp.Unet(encoder_name="resnet50", in_channels=3, classes=1).to(device=DEVICE)

	test_loader = get_test_loader(
		TEST_IMG_DIR,
		TEST_MASK_DIR,
		BATCH_SIZE,
		test_transforms,
		NUM_WORKERS,
		PIN_MEMORY,
	)

	load_checkpoint(torch.load("saved_models/unet_resnet50/imagenet_weights/no_preproc/0709_e_4_j_0.8281_d_0.9033_a_0.9086.pth.tar"), model)
	check_performance(test_loader, model, device=DEVICE)

	# print some examples to a folder
	save_predictions_as_imgs(test_loader, model, folder="pred_samples/", device=DEVICE)

if __name__ == "__main__":
	torch.cuda.set_device(7)
	main()