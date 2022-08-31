import math

import torch
from torch import nn
import torch.nn.functional as F


def ConvReluConv(in_channel, conv_channels, out_channel, bias_fill=False, bias_value=0.0):
	""" Userful for Head output"""
	feat_conv = nn.Conv2d(in_channel, conv_channels, kernel_size=3, padding=1, bias=True)
	relu = nn.ReLU()
	out_conv = nn.Conv2d(conv_channels, out_channel, kernel_size=1, stride=1, padding=0)
	if bias_fill:
		out_conv.bias.data.fill_(bias_value)

	return nn.Sequential(feat_conv, relu, out_conv)

def make_conv(in_channels, out_channels, conv_type="normal", kernel_size=3, padding=None, stride=1,
              depth_multiplier=1, **kwargs):
	"""Create a convolution layer. Options: deformable, separable, or normal convolution
	"""
	assert conv_type in ("separable", "normal")
	if padding is None:
		padding = (kernel_size - 1) // 2

	if conv_type == "separable":
		hidden_channels = in_channels * depth_multiplier
		conv_layer = nn.Sequential(
			# dw
			nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding, stride=stride,
			          groups=in_channels, bias=False),
			nn.BatchNorm2d(in_channels),
			nn.ReLU6(inplace=True),
			# pw
			nn.Conv2d(hidden_channels, out_channels, 1, bias=False, stride=stride),
			nn.BatchNorm2d(out_channels),
			nn.ReLU6(inplace=True)
		)
		nn.init.kaiming_normal_(conv_layer[0].weight, mode="fan_out", nonlinearity="relu")
		nn.init.kaiming_normal_(conv_layer[3].weight, mode="fan_out", nonlinearity="relu")

	else:  # normal convolution
		conv_layer = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)
		nn.init.kaiming_normal_(conv_layer[0].weight, mode="fan_out", nonlinearity="relu")

	return conv_layer


def make_upsample(upsample_type="nearest", deconv_channels=None, deconv_kernel=4, deconv_init_bilinear=True, **kwargs):
	"""Create an upsample layer. Options: convolution transpose, bilinear upsampling, or nearest upsampling
	"""
	assert upsample_type in ("conv_transpose", "bilinear", "nearest")

	if upsample_type == "conv_transpose":
		output_padding = deconv_kernel % 2
		padding = (deconv_kernel + output_padding) // 2 - 1

		upsample = nn.ConvTranspose2d(deconv_channels, deconv_channels, deconv_kernel, stride=2, padding=padding,
		                              output_padding=output_padding, bias=False)
		bn = nn.BatchNorm2d(deconv_channels)
		relu = nn.ReLU(inplace=True)
		upsample_layer = nn.Sequential(upsample, bn, relu)

		if deconv_init_bilinear:  # TF CenterNet does not do this
			_init_bilinear_upsampling(upsample)

	else:
		upsample_layer = nn.Upsample(scale_factor=2, mode=upsample_type)

	return upsample_layer


def _init_bilinear_upsampling(deconv_layer):
	"""Initialize convolution transpose layer as bilinear upsampling to help with training stability
	"""
	# https://github.com/ucbdrive/dla/blob/master/dla_up.py#L26-L33
	w = deconv_layer.weight.data
	f = math.ceil(w.size(2) / 2)
	c = (2 * f - 1 - f % 2) / (f * 2.)

	for i in range(w.size(2)):
		for j in range(w.size(3)):
			w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))

	for c in range(1, w.size(0)):
		w[c, 0, :, :] = w[0, 0, :, :]


def make_downsample(downsample_type="max", conv_channels=None, conv_kernel=3, **kwargs):
	"""Create a downsample layer. Options: convolution, max pooling, or average pooling
	"""
	assert downsample_type in ("max", "average", "conv")

	if downsample_type == "conv":
		downsample = nn.Conv2d(conv_channels, conv_channels, conv_kernel, stride=2, padding="same", bias=False)
		bn = nn.BatchNorm2d(conv_channels)
		relu = nn.ReLU(inplace=True)
		downsample_layer = nn.Sequential(downsample, bn, relu)

		nn.init.kaiming_normal_(downsample.weight, mode="fan_out", nonlinearity="relu")

	elif downsample_type == "max":
		downsample_layer = nn.MaxPool2d(2, 2)
	else:
		downsample_layer = nn.AvgPool2d(2, 2)

	return downsample_layer


class Fuse(nn.Module):
	"""Fusion node to be used for feature fusion. To be used in `BiFPNNeck` and `IDANeck`. The last input will be resized.

	Formula
		no weight: out = conv(in1 + resize(in2))
		weighted: out = conv((in1*w1 + resize(in2)*w2) / (w1 + w2 + eps))
	"""

	def __init__(self, num_fused, out, resize, upsample="nearest", downsample="max", conv_type="normal",
	             weighted_fusion=False):
		super().__init__()
		assert resize in ("up", "down")
		assert num_fused >= 2

		self.weighted_fusion = weighted_fusion
		self.num_fused = num_fused
		if weighted_fusion:
			self.weights = nn.Parameter(torch.ones(num_fused), requires_grad=True)

		if resize == "up":
			self.resize = make_upsample(upsample_type=upsample, deconv_channels=out)
		else:
			self.resize = make_downsample(downsample=downsample, conv_channels=out)

		self.output_conv = make_conv(out, out, conv_type=conv_type)

	def forward(self, *features, eps=1e-6):

		last = self.resize(features[-1])

		if self.weighted_fusion:
			weights = F.relu(self.weights)
			weights = weights / (torch.sum(weights) + eps)
			out = features[0] * weights[0]
			for i in range(1, self.num_fused-1):
				out = out + (features[i] * weights[i])
			out = out + (last * weights[-1])
		else:
			out = features[0]
			for i in range(1, self.num_fused-1):
				out = out + features[i]
			out = out + last

		out = self.output_conv(out)
		return out




