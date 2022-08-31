# -*- coding: utf-8 -*-
# ---------------------

import torch
import numpy as np

from .heatmaps import kps_to_heatmaps
from .heatmaps import CornerNetRadius, FixedRadius
from .cls import DET_CLS, WTR_CLS, TD_CLS, SN_CLS


class DetProcessor:

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

		self.sigma = CornerNetRadius()

	def bounding_boxes(self, annot):
		# Preprocess label (boxes)
		labels = annot.get("labels", None)
		if labels is not None:
			boxes = torch.stack([torch.tensor([*l['box2d'].values()]) for l in labels])  # x1,y1,x2,y2
			classes = [DET_CLS[l['category']] for l in labels]
			occlusion = [l['attributes']['occluded'] for l in labels]

			# Remove 'other vehicle'
			io = [i for i, v in enumerate(classes) if v >= 10]
			boxes = [k for i, k in enumerate(boxes) if i not in io]
			classes = [c for i, c in enumerate(classes) if i not in io]
			occlusion = [o for i, o in enumerate(occlusion) if i not in io]

		else:
			boxes = []
			classes = []
			occlusion = []

		return labels, boxes, classes, occlusion

	def scene_classification(self, annot):
		attrs = annot.get("attributes", None)
		cls = {}

		if attrs is not None:
			cls["weather"] = WTR_CLS[attrs["weather"]]
			cls["scene"] = SN_CLS[attrs["scene"]]
			cls["timeofday"] = TD_CLS[attrs["timeofday"]]

		return cls

	def targets(self, labels, bboxes, classes):
		if labels is not None and len(bboxes) > 0:

			# Obtain box centers in output space
			boxes_pt = torch.tensor(bboxes) / self.output_s
			boxes_cwh = self.xyxy2cxcywh(boxes_pt)
			radii = torch.tensor([self.sigma(w, h) for w, h in boxes_cwh[..., 2:] * self.output_s])

			# Clip and round
			centers = boxes_cwh[:, :2]
			centers[:, 0] = torch.clip(centers[:, 0], 0, self.target_w - 1)
			centers[:, 1] = torch.clip(centers[:, 1], 0, self.target_h - 1)
			centers = torch.round(centers)

			assert centers[:, 0].max() < self.target_w and centers[:, 1].max() < self.target_h  # <-- shit here
			assert centers[:, 0].min() >= 0 and centers[:, 1].min() >= 0

			# Compute target heatmaps
			kp_cls = [[] for _ in range(self.classes)]
			for ic, c in enumerate(classes):
				kp_cls[c].append(torch.cat([centers[ic], radii[ic].unsqueeze(0)]))  # cx, cy, sigma

			kp_cls = [torch.stack(t) if len(t) > 0 else torch.tensor([]) for t in kp_cls]

			# Generate target (Heatmap)
			heatmap = kps_to_heatmaps(kp_cls, self.target_w, self.target_h, sigma=None)

			# Compute target offsets
			ofs_x = boxes_pt[..., 0::2] - centers[..., 0].unsqueeze(-1)
			ofs_y = boxes_pt[..., 1::2] - centers[..., 1].unsqueeze(-1)
			ofs = torch.cat([ofs_x, ofs_y], dim=-1)  # (x1-cx, x2-cx), (y1-cy, y2-cy)

		else:
			heatmap = torch.zeros((self.classes, self.target_h, self.target_w), dtype=torch.float32)
			centers = torch.zeros((0, 2), dtype=torch.int)
			ofs = torch.zeros((0, 4), dtype=torch.float32)

		return heatmap, centers, ofs

	@staticmethod
	def xyxy2cxcywh(boxes):
		w = (boxes[:, 2] - boxes[:, 0])
		h = (boxes[:, 3] - boxes[:, 1])
		cx = boxes[:, 0] + w / 2
		cy = boxes[:, 1] + h / 2

		return torch.stack([cx, cy, w, h], dim=1)
