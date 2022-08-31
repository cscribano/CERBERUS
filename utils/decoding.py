# -*- coding: utf-8 -*-
# ---------------------

from typing import List, Tuple
import torch

class PseudoNMS(torch.nn.Module):
	def __init__(self, nms_kernels):
		# type: (List[Tuple[int, int]]) -> None

		super().__init__()
		
		pooling = []
		for k in nms_kernels:
			padding = ((k[0] - 1) // 2, (k[1] - 1) // 2)
			pool = torch.nn.MaxPool2d(kernel_size=k, stride=1, padding=padding)
			pooling.append(pool)
			
		self.pooling = torch.nn.ModuleList(pooling)

	def forward(self, heatmap):

		masks = []
		for pool in self.pooling:
			nms_mask = pool(heatmap)
			nms_mask = (nms_mask == heatmap)
			masks.append(nms_mask)

		for mask in masks:
			heatmap = heatmap * mask

		return heatmap


def kp_from_heatmap(heatmap, th, nms_kernel=3, pseudo_nms=True):

	# 1. pseudo-nms via max pool
	if pseudo_nms:
		padding = (nms_kernel - 1) // 2
		mask = torch.nn.functional.max_pool2d(heatmap, kernel_size=nms_kernel, stride=1, padding=padding) == heatmap
		heatmap = heatmap * mask

	# Get best candidate at each heatmap location, since box regression is shared
	heatmap, labels = torch.max(heatmap, dim=1)

	# Flatten and get values
	indices = torch.nonzero(heatmap.gt(th), as_tuple=False).flip(1)
	scores = heatmap[0, indices[:, 1], indices[:, 0]]
	labels = labels[0, indices[:, 1], indices[:, 0]]

	return scores, indices, labels
