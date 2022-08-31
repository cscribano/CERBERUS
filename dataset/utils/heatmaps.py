# -*- coding: utf-8 -*-
# ---------------------

import numpy as np
import torch

class FixedRadius:
	def __init__(self, r: float = 1.):
		self.r = r

	def __call__(self, w, h):
		return self.r#, self.r

class CornerNetRadius:
	def __init__(self, min_overlap: float = 0.7):
		self.min_overlap = min_overlap

	# Explanation: https://github.com/princeton-vl/CornerNet/issues/110
	# Source: https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
	def __call__(self, width, height):
		a1 = 1
		b1 = (height + width)
		c1 = width * height * (1 - self.min_overlap) / (1 + self.min_overlap)
		sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
		r1 = (b1 + sq1) / 2

		a2 = 4
		b2 = 2 * (height + width)
		c2 = (1 - self.min_overlap) * width * height
		sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
		r2 = (b2 + sq2) / 2

		a3 = 4 * self.min_overlap
		b3 = -2 * self.min_overlap * (height + width)
		c3 = (self.min_overlap - 1) * width * height
		sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
		r3 = (b3 + sq3) / 2
		return max(min(r1, r2, r3) / 6, 2)

def kps_to_heatmaps(annotation, w, h, sigma=None):

	heatmaps_list = []

	for cls_keypoints in annotation:

		# generate heatmap from list of (x, y, z) coordinates
		# retrieve one (W,H) heatmap for each keypoint
		if len(cls_keypoints) != 0:
			# Normalize coordinates
			# cls_keypoints = torch.tensor(cls_keypoints) / torch.tensor([IMG_HEIGHT, IMG_WIDTH])

			# Generate heatmap
			if sigma is None:
				assert cls_keypoints.shape[-1] == 3
				kern = make_gkern_2d(h, w, None)
				heatmaps = torch.stack([kern(x, s) for x, s in zip(cls_keypoints[..., :2],
				                                                   cls_keypoints[..., -1])], dim=0)
			else:
				assert cls_keypoints.shape[-1] == 2
				kern = make_gkern_2d(h, w, sigma)
				heatmaps = torch.stack([kern(x) for x in cls_keypoints], dim=0)
		else:
			heatmaps = torch.zeros(1, h, w)

		# Combine individual heatmaps in a single tensor
		heatmap = torch.max(heatmaps, dim=0)[0]
		heatmaps_list.append(heatmap)

	# Combine keypoints heatmaps in a single tensor
	total_heatmap = torch.stack(heatmaps_list, 0)

	return total_heatmap

def make_gkern_2d(h, w, s=None, device='cpu'):
	if s is None:
		def gk(x, s):
			return gkern_2d(h, w, x, s, device=device)
	else:
		def gk(x):
			return gkern_2d(h, w, x, s, device=device)

	return gk

def gkern_2d(h, w, center, s, device='cuda'):
	# type: (int, int, Tuple[int, int], float, str) -> torch.Tensor
	"""
	:param h: heatmap image height
	:param w: heatmap image width
	:param center: Gaussian center (x,y,z)
	:param s: Gaussian sigma
	:param device: 'cuda' or 'cpu' -> device used do compute heatmaps
	:return: Torch tensor with shape (h, w, d) with A Gaussian centered in `center`
	"""

	x = torch.arange(0, w, 1).type('torch.FloatTensor').to(device)
	y = torch.arange(0, h, 1).type('torch.FloatTensor').to(device)

	y = y.unsqueeze(1)

	x0 = center[0]  # * w
	y0 = center[1]  # * h

	g = torch.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2) / s ** 2)

	return g
