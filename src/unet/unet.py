import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop

class double_conv(nn.Module):
	"""(3x3 conv, ReLU) * 2"""
	def __init__(self, in_channels, out_channels, kernel_size, stride):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)
		return x

class UNet(nn.Module):

	def __init__(self, n_channels, n_classes):
		super().__init__()

		self.double_conv_1 = double_conv(n_channels, 64, kernel_size=3, stride=1)
		self.double_conv_2 = double_conv(64, 128, kernel_size=3, stride=1)
		self.double_conv_3 = double_conv(128, 256, kernel_size=3, stride=1)
		self.double_conv_4 = double_conv(256, 512, kernel_size=3, stride=1)
		self.double_conv_5 = double_conv(512, 1024, kernel_size=3, stride=1)

		self.double_conv_6 = double_conv(1024, 512, kernel_size=3, stride=1)
		self.double_conv_7 = double_conv(512, 256, kernel_size=3, stride=1)
		self.double_conv_8 = double_conv(256, 128, kernel_size=3, stride=1)
		self.double_conv_9 = double_conv(128, 64, kernel_size=3, stride=1)

		self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

		self.up_conv_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
		self.up_conv_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
		self.up_conv_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
		self.up_conv_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

		self.last_conv = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

	def forward(self, x):
		conv_1 = self.double_conv_1(x)		# n_channels -> 64 channels
		x = self.max_pool(conv_1)

		conv_2 = self.double_conv_2(x)		# 64 -> 128 channels
		x = self.max_pool(conv_2)

		conv_3 = self.double_conv_3(x)		# 128 -> 256 channels
		x = self.max_pool(conv_3)

		conv_4 = self.double_conv_4(x)		# 256 -> 512 channels
		x = self.max_pool(conv_4)

		x = self.double_conv_5(x)			# 512 -> 1024 channels

		x = self.up_conv_1(x)				# 1024 -> 512 channels
		# sCHW
		conv_4_c = CenterCrop((x.shape[2], x.shape[3]))(conv_4)
		x = torch.cat([x, conv_4_c], dim=1)	# 512 + 512 = 1024 channels
		x = self.double_conv_6(x)			# 1024 -> 512 channels
		x = self.up_conv_2(x)				# 512 -> 256 channels

		conv_3_c = CenterCrop((x.shape[2], x.shape[3]))(conv_3)
		x = torch.cat([x, conv_3_c], dim=1)	# 256 + 256 = 512 channels
		x = self.double_conv_7(x)			# 512 -> 256 channels
		x = self.up_conv_3(x)				# 256 -> 128 channels

		conv_2_c = CenterCrop((x.shape[2], x.shape[3]))(conv_2)
		x = torch.cat([x, conv_2_c], dim=1)	# 128 + 128 = 256 channels
		x = self.double_conv_8(x)			# 256 -> 128 channels
		x = self.up_conv_4(x)				# 128 -> 64 channels

		conv_1_c = CenterCrop((x.shape[2], x.shape[3]))(conv_1)
		x = torch.cat([x, conv_1_c], dim=1)	# 64 + 64 = 128 channels
		x = self.double_conv_9(x)			# 128 -> 64 channels
		out = self.last_conv(x)				# 64 -> n_classes channels

		return out