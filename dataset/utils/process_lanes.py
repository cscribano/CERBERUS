# -*- coding: utf-8 -*-
# ---------------------
import torch
import numpy as np

from dataset.utils.heatmaps import kps_to_heatmaps
from dataset.utils.cls import LANE_CLS

class LaneProcessor:

	def __init__(self, classes, output_s, target_w, target_h):
		# type: (int, int, int, int) -> None
		"""
		:param classes: Number of lane classes
		:param output_s: Output stride wrt input shape
		:param target_w: output width (output_s * input width)
		:param target_h: output height (output_s * input height)
		"""

		self.classes = classes
		self.output_s = output_s
		self.target_w = target_w
		self.target_h = target_h

	def keypoints(self, annot):
		# type: (dict) -> (list, list, list, list)
		"""
		:param annot:
		:return:
		"""

		labels = annot.get("labels", None)
		if labels is not None:
			labels = [l for l in labels if l is not None]
			if len(labels) == 0:
				return labels, [], [], []

			keypoints = torch.cat([torch.tensor(l['keypoints']) for l in labels])
			assert len(keypoints.shape) == 2 and keypoints.shape[0] >= 1

			lenghts = [len(l['keypoints']) for l in labels]
			cls = [LANE_CLS[l['category']] for i, l in enumerate(labels) for _ in range(lenghts[i])]
			ids = [int(l['id']) for i, l in enumerate(labels) for _ in range(lenghts[i])]

			# Remove non visible keypoints
			visible = torch.stack([keypoints[:, 0].ceil() < 1280, keypoints[:, 1].ceil() < 720,
			                       keypoints[:, 0].floor() >= 0, keypoints[:, 1].floor() >= 0])
			assert len(visible.shape) == 2 and visible.shape[0] >= 1
			visible = visible.min(dim=0)[0]

			keypoints = keypoints[visible.nonzero().squeeze(1)].tolist()
			classes = [cls[c] for c in visible.nonzero().squeeze(1)]
			ids = [ids[c] for c in visible.nonzero().squeeze(1)]

			return labels, keypoints, classes, ids

		return labels, [], [], []

	def targets(self, labels, keypoints, classes, ids):

		if labels is not None and len(keypoints) > 0:

			all_ids = set(ids)

			# Clip and round
			keypoints = torch.tensor(keypoints) / self.output_s
			centers = keypoints.clone()

			centers[:, 0] = torch.clip(centers[:, 0], 0, self.target_w - 1)
			centers[:, 1] = torch.clip(centers[:, 1], 0, self.target_h - 1)
			centers = torch.round(
				centers)

			assert centers[:, 0].max() < self.target_w and centers[:, 1].max() < self.target_h
			assert centers[:, 0].min() >= 0 and centers[:, 1].min() >= 0

			# Generate target (Heatmap)
			kp_cls = [[] for _ in range(self.classes)]
			for ic, c in enumerate(classes):
				# kp_cls[c].append(centers[ic]) #<--- to enable rounding
				kp_cls[c].append(keypoints[ic])

			kp_cls = [torch.stack(t) if len(t) > 0 else torch.tensor([]) for t in kp_cls]
			heatmap = kps_to_heatmaps(kp_cls, self.target_w, self.target_h, sigma=2)

			# Generate dequantizzation offsets
			quant_offsets = (keypoints - centers).to(torch.float32)

			# Group keypoints belonging to the same lane
			lane_ids = torch.tensor(ids)
			lanes_kp = [keypoints[lane_ids.eq(i).nonzero().squeeze(1)] for i in all_ids]
			centers = [centers[lane_ids.eq(i).nonzero().squeeze(1)] for i in all_ids]
			quant_offsets = [quant_offsets[lane_ids.eq(i).nonzero().squeeze(1)] for i in all_ids]

			# Generate offsets to lane center
			l_offsets = [b[len(b) // 2] - b for b in centers]

			# Flatten
			centers = torch.cat(centers)
			center_offsets = torch.cat(l_offsets)
			quant_offsets = torch.cat(quant_offsets)

		else:
			heatmap = torch.zeros((self.classes, self.target_h, self.target_w), dtype=torch.float32)
			centers = torch.zeros((0, 2), dtype=torch.int)
			lanes_kp = torch.zeros((0, 2), dtype=torch.float32)
			center_offsets = torch.zeros((0, 2), dtype=torch.float32)
			quant_offsets = torch.zeros((0, 2), dtype=torch.float32)

		return heatmap, centers, center_offsets, quant_offsets, lanes_kp
