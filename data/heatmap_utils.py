# -*- coding: utf-8 -*-
# ---------------------

import torch

def kps_to_heatmaps(annotation, w, h, sigma):

	heatmaps_list = []

	for cls_keypoints in annotation:

		# generate heatmap from list of (x, y, z) coordinates
		# retrieve one (W,H) heatmap for each keypoint
		if len(cls_keypoints) != 0:
			# Normalize coordinates
			# cls_keypoints = torch.tensor(cls_keypoints) / torch.tensor([IMG_HEIGHT, IMG_WIDTH])

			# Generate heatmap
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

def make_gkern_2d(h, w, s, device='cpu'):
	def gk(head):
		return gkern_2d(h, w, head, s, device=device)

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

	x0 = center[0] * w
	y0 = center[1] * h

	return torch.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2) / s ** 2)
