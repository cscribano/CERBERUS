import warnings
from typing import Callable, Any, Optional, List, Union

import torch
from torch import Tensor
from torch import nn

from .misc import ConvNormActivation
from .misc import _make_divisible
from collections import OrderedDict

import warnings
from torch.hub import load_state_dict_from_url

__all__ = ["MobileNetV2", "mobilenet_v2"]


model_urls = {
	"mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}


# necessary for backwards compatibility
class _DeprecatedConvBNAct(ConvNormActivation):
	def __init__(self, *args, **kwargs):
		warnings.warn(
			"The ConvBNReLU/ConvBNActivation classes are deprecated since 0.12 and will be removed in 0.14. "
			"Use torchvision.ops.misc.ConvNormActivation instead.",
			FutureWarning,
		)
		if kwargs.get("norm_layer", None) is None:
			kwargs["norm_layer"] = nn.BatchNorm2d
		if kwargs.get("activation_layer", None) is None:
			kwargs["activation_layer"] = nn.ReLU6
		super().__init__(*args, **kwargs)


ConvBNReLU = _DeprecatedConvBNAct
ConvBNActivation = _DeprecatedConvBNAct


class InvertedResidual(nn.Module):
	def __init__(
			self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
	) -> None:
		super().__init__()
		self.stride = stride
		assert stride in [1, 2]

		if norm_layer is None:
			norm_layer = nn.BatchNorm2d

		hidden_dim = int(round(inp * expand_ratio))
		self.use_res_connect = self.stride == 1 and inp == oup

		layers: List[nn.Module] = []
		if expand_ratio != 1:
			# pw
			layers.append(
				ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
			)
		layers.extend(
			[
				# dw
				ConvNormActivation(
					hidden_dim,
					hidden_dim,
					stride=stride,
					groups=hidden_dim,
					norm_layer=norm_layer,
					activation_layer=nn.ReLU6,
				),
				# pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				norm_layer(oup),
			]
		)
		self.conv = nn.Sequential(*layers)
		self.out_channels = oup
		self._is_cn = stride > 1

	def forward(self, x: Tensor) -> Tensor:
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)


class MobileNetV2(nn.Module):
	def __init__(
			self,
			num_classes: int = 1000,
			width_mult: float = 1.0,
			inverted_residual_setting: Optional[List[List[int]]] = None,
			round_nearest: int = 8,
			block: Optional[Callable[..., nn.Module]] = None,
			norm_layer: Optional[Callable[..., nn.Module]] = None,
			dropout: float = 0.2,
	) -> None:
		"""
		MobileNet V2 main class

		Args:
			num_classes (int): Number of classes
			width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
			inverted_residual_setting: Network structure
			round_nearest (int): Round the number of channels in each layer to be a multiple of this number
			Set to 1 to turn off rounding
			block: Module specifying inverted residual building block for mobilenet
			norm_layer: Module specifying the normalization layer to use
			dropout (float): The droupout probability

		"""
		super().__init__()

		if block is None:
			block = InvertedResidual

		if norm_layer is None:
			norm_layer = nn.BatchNorm2d

		input_channel = 32
		last_channel = 1280

		if inverted_residual_setting is None:
			inverted_residual_setting = [
				# t, c, n, s
				[1, 16, 1, 1],
				[6, 24, 2, 2],
				[6, 32, 3, 2],
				[6, 64, 4, 2],
				[6, 96, 3, 1],
				[6, 160, 3, 2],
				[6, 320, 1, 1],
			]

		# only check the first element, assuming user knows t,c,n,s are required
		if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
			raise ValueError(
				f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
			)

		# building first layer
		outplanes = [input_channel]
		input_channel = _make_divisible(input_channel * width_mult, round_nearest)
		self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
		features: List[nn.Module] = [
			ConvNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
		]
		# building inverted residual blocks
		for t, c, n, s in inverted_residual_setting:
			output_channel = _make_divisible(c * width_mult, round_nearest)
			for i in range(n):
				stride = s if i == 0 else 1
				features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
				input_channel = output_channel
				outplanes.append(output_channel)

		"""
		# building last several layers
		features.append(
			ConvNormActivation(
				input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
			)
		)
		"""
		# make it nn.Sequential
		self.features = nn.ModuleList(features)

		self.gates = [4, 7, 14, 18]
		self.outplanes = [outplanes[g-1] for g in self.gates]

		# weight initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out")
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.zeros_(m.bias)

	def _forward_impl(self, x: Tensor) -> Union[Tensor, Any]:

		for n in range(0, self.gates[0]):
			x = self.features[n](x)
		x1 = x

		for n in range(self.gates[0], self.gates[1]):
			x = self.features[n](x)
		x2 = x

		for n in range(self.gates[1], self.gates[2]):
			x = self.features[n](x)
		x3 = x

		for n in range(self.gates[2], self.gates[3]):
			x = self.features[n](x)

		return x1, x2, x3, x

	def forward(self, x: Tensor) -> Union[Tensor, Any]:
		return self._forward_impl(x)


def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
	"""
	Constructs a MobileNetV2 architecture from
	`"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	model = MobileNetV2(**kwargs)
	if pretrained:
		arch = "mobilenet_v2"
		if model_urls.get(arch, None) is None:
			raise ValueError(f"No checkpoint is available for model type {arch}")
		state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

		# Discard removed layers
		model_dict = model.state_dict()
		matched_layers, discarded_layers = [], []
		new_state_dict = OrderedDict()

		for k, v in state_dict.items():

			if k in model_dict and model_dict[k].size() == v.size():
				new_state_dict[k] = v
				matched_layers.append(k)
			else:
				discarded_layers.append(k)

		model_dict.update(new_state_dict)

		if len(matched_layers) == 0:
			warnings.warn(
				'The pretrained weights for "{}" cannot be loaded, '
				'please check the key names manually '
				'(** ignored and continue **)'.format(arch)
			)
		else:
			print(
				'Successfully loaded imagenet pretrained weights for "{}"'.
				format(arch)
			)
			if len(discarded_layers) > 0:
				print(
					'** The following layers are discarded '
					'due to unmatched keys or layer size: {}'.
					format(discarded_layers)
				)

		model.load_state_dict(model_dict)

	return model

if __name__ == '__main__':
	m = mobilenet_v2(pretrained=True)
	x = torch.rand((1,3,320,640), dtype=torch.float32)
	y = m(x)
	print(m.outplanes)

	"""
	0 torch.Size([1, 32, 160, 320])
	1 torch.Size([1, 16, 160, 320])
	2 torch.Size([1, 24, 80, 160])
	3 torch.Size([1, 24, 80, 160])
	4 torch.Size([1, 32, 40, 80])
	5 torch.Size([1, 32, 40, 80])
	6 torch.Size([1, 32, 40, 80])
	7 torch.Size([1, 64, 20, 40])
	8 torch.Size([1, 64, 20, 40])
	9 torch.Size([1, 64, 20, 40])
	10 torch.Size([1, 64, 20, 40])
	11 torch.Size([1, 96, 20, 40])
	12 torch.Size([1, 96, 20, 40])
	13 torch.Size([1, 96, 20, 40])
	14 torch.Size([1, 160, 10, 20])
	15 torch.Size([1, 160, 10, 20])
	16 torch.Size([1, 160, 10, 20])
	17 torch.Size([1, 320, 10, 20])
	
	[24, 32, 96, 320]
	
	"""
