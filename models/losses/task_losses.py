# -*- coding: utf-8 -*-
# ---------------------

import torch
from torch import nn
import torch.nn.functional as F
from models.losses.heatmap_loss import *

from conf import Conf

class LanesLoss(nn.Module):

	def __init__(self, cnf):
		# type: (Conf) -> ()

		super().__init__()
		self.cnf = cnf
		self.q_offsets = cnf.base.get("lane_q_offsets", False)

		heatmap_loss = self.cnf.loss.heatmap_loss.get("name", "nn.MSELoss")
		self.heatmap_loss = eval(heatmap_loss)(**self.cnf.loss.heatmap_loss.args)
		self.offset_loss = nn.L1Loss()

	def forward(self, preds, targets):
		# type: (dict[torch.tensor, ...], dict[torch.tensor, ...]) -> torch.tensor

		hm_true, kp_true, ofs_true, q_ofs = targets["heatmaps"], targets["keypoints"], \
		                                    targets["offsets"], targets["quant_offsets"]

		hm_pred, ofs_pred = preds["heatmaps"], preds["offsets"]

		# Heatmap
		hm_loss = self.heatmap_loss(hm_pred, hm_true)

		# Embeddings
		b_idx = torch.tensor([i for i, b in enumerate(kp_true) for _ in range(b.shape[0])])
		kp_true = torch.cat(kp_true).long()

		embs_pred = ofs_pred[b_idx, :, kp_true[:, 1], kp_true[:, 0]]
		embs_true = torch.cat(ofs_true)

		embd_loss = self.offset_loss(embs_pred, embs_true) * 0.8  # 0.4

		# Dequantizzation offsets
		quant_loss = torch.tensor(0, device=hm_true.device)
		if self.q_offsets:
			q_pred = targets["quant"]
			q_pred = q_pred[b_idx, :, kp_true[:, 1], kp_true[:, 0]]
			q_ofs = torch.cat(q_ofs)

			quant_loss = self.offset_loss(q_pred, q_ofs)

		return embd_loss + hm_loss + quant_loss, {"l_heat": hm_loss.item(),
		                                          "l_emb": embd_loss.item(), "l_quant": quant_loss.item()}

class ObjectsLoss(nn.Module):

	def __init__(self, cnf):
		# type: (Conf) -> ()

		super().__init__()
		self.cnf = cnf
		self.occlusion = cnf.base.get("occlusion_cls", True)

		# Task specific losses
		heatmap_loss = self.cnf.loss.heatmap_loss.get("name", "nn.MSELoss")
		self.heatmap_loss = eval(heatmap_loss)(**self.cnf.loss.heatmap_loss.args)
		self.offset_loss = nn.L1Loss()

	def forward(self, preds, targets):
		# type: (dict[torch.tensor, ...], dict[torch.tensor, ...]) -> torch.tensor

		hm_true, oc_true, ofs_true, ocl_true = targets["heatmaps"], targets["centers"], \
		                                       targets["offsets"], targets["occlusion"]

		hm_pred, ofs_pred, ocl_pred = preds["heatmaps"], preds["offsets"], preds["occlusion"]

		# Heatmap
		hm_loss = self.heatmap_loss(hm_pred, hm_true)

		# xxyy offsets
		# (x1-cx, x2-cx), (y1-cy, y2-cy)
		b_idx = torch.tensor([i for i, b in enumerate(oc_true) for _ in range(b.shape[0])])
		oc_true = torch.cat(oc_true).long()

		ofs_pred = ofs_pred[b_idx, :, oc_true[:, 1], oc_true[:, 0]]
		ofs_true = torch.cat(ofs_true)

		ofs_loss = self.offset_loss(ofs_pred, ofs_true)

		# Occlusion classification
		if self.occlusion:
			ocl_pred = ocl_pred[b_idx, :, oc_true[:, 1], oc_true[:, 0]]
			ocl_true = torch.cat(ocl_true)
			ocl_loss = F.binary_cross_entropy_with_logits(ocl_pred.squeeze(-1), ocl_true)  #* 0.5
		else:
			ocl_loss = torch.tensor(0.0, device=hm_pred.device)


		return hm_loss + ofs_loss + ocl_loss, {"d_heat": hm_loss.item(),
		                                       "d_ofs": ofs_loss.item(), "d_ocl": ocl_loss.item()}

class ClsLoss(nn.Module):

	def __init__(self, cnf):
		# type: (Conf) -> ()

		super().__init__()
		self.cnf = cnf

		scn_loss = self.cnf.loss.scn_loss.get("name", "nn.CrossEntropyLoss")
		self.scn_loss = eval(scn_loss)(**self.cnf.loss.scn_loss.args)

	def forward(self, preds, targets):
		# type: (dict[torch.tensor, ...], dict[torch.tensor, ...]) -> torch.tensor

		# weather, scene, timeofday
		w_pred = preds["weather"]
		s_pred = preds["scene"]
		t_pred = preds["timeofday"]

		s1 = self.scn_loss(w_pred, targets["weather"])
		s2 = self.scn_loss(s_pred, targets["scene"])
		s3 = self.scn_loss(t_pred, targets["timeofday"])

		scn_loss = (s1 + s2 + s3) * 0.1

		return scn_loss, {"scn": scn_loss.item()}
