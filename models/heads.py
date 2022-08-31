# -*- coding: utf-8 -*-
# ---------------------

import torch
from torch import nn
from abc import ABCMeta, abstractmethod

from utils.decoding import PseudoNMS
from .layers import make_conv, ConvReluConv

class BaseHead(nn.Module, metaclass=ABCMeta):

	def __init__(self):
		# type: () -> None
		super().__init__()

	def forward(self, x, decode):
		# type: (torch.Tensor, bool) -> dict[str, torch.Tensor, ...]
		...

class ObjectHead(BaseHead):

	def __init__(self, num_classes=80, in_channels=256, conv_channels=64):

		super(ObjectHead, self).__init__()
		self.cls_head = ConvReluConv(in_channels, conv_channels, num_classes, bias_fill=True, bias_value=-4.6)
		self.ofs_out = ConvReluConv(in_channels, in_channels, 4)
		self.occl = ConvReluConv(in_channels, in_channels, 1)

		self.nms = PseudoNMS(nms_kernels=[(3, 3)])

	def forward(self, x, nms=False):
		hm = self.cls_head(x).sigmoid()
		wh = self.ofs_out(x)
		oc = self.occl(x)

		if nms:
			hm = self.nms(hm)

		ret = {
			"heatmaps": hm,
			"offsets": wh,
			"occlusion": oc
		}

		return ret

class LaneHead(BaseHead):

	def __init__(self, num_classes=80, in_channels=256, quant_offsets=False, conv_channels=64):
		super(LaneHead, self).__init__()
		self.cls_head = ConvReluConv(in_channels, conv_channels, num_classes, bias_fill=True, bias_value=-4.6)
		self.emb_out = ConvReluConv(in_channels, in_channels, 2, bias_fill=True, bias_value=0.1)

		# Dequantizzation offsets
		self.quant_offsets = quant_offsets
		if quant_offsets:
			self.quant_out = ConvReluConv(in_channels, in_channels, 2, bias_fill=True, bias_value=0.1)

		self.nms = PseudoNMS(nms_kernels=[(1, 3), (3, 1)])

	def forward(self, x, nms=False):
		hm = self.cls_head(x).sigmoid()
		emb = self.emb_out(x)

		if nms:
			hm = self.nms(hm)

		ret = {
			"heatmaps": hm,
			"offsets": emb,
		}

		if self.quant_offsets:
			quant = self.quant_out(x)
			ret["quant"] = quant

		return ret

class ScnHead(BaseHead):
	def __init__(self, classes, in_channels):
		super(ScnHead, self).__init__()

		self.cls_splits = classes

		self.c1 = make_conv(in_channels, 64)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(64, sum(classes))

	def forward(self, x, argmax=False):
		x = self.c1(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)

		w_pred, s_pred, t_pred = torch.split(x, self.cls_splits, 1)

		if argmax:
			w_pred = w_pred.argmax(-1)
			s_pred = s_pred.argmax(-1)
			t_pred = t_pred.argmax(-1)

		ret = {
			"weather": w_pred,
			"scene": s_pred,
			"timeofday": t_pred
		}

		return ret
