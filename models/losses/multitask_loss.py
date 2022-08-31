# -*- coding: utf-8 -*-
# ---------------------

import torch
from torch import nn
from models.losses.task_losses import ObjectsLoss, LanesLoss, ClsLoss
from models.losses.heatmap_loss import *

from conf import Conf

class MultiTaskLoss(nn.Module):

	def __init__(self, cnf):
		# type: (Conf) -> ()

		super().__init__()
		self.cnf = cnf

		# Configuration
		self.lane_det = cnf.base.get("lane_det", True)
		self.obj_det = cnf.base.get("object_det", True)
		self.scene_cls = cnf.base.get("scene_cls", False)

		# Task specific losses
		self.lanes_loss = LanesLoss(cnf)
		self.objects_loss = ObjectsLoss(cnf)
		self.cls_loss = ClsLoss(cnf)

	def forward(self, preds, targets):
		# type: (dict[torch.tensor, ...], dict[torch.tensor, ...]) -> torch.tensor

		l_loss, d_loss, scn_loss = 0.0, 0.0, 0.0
		l_detail, d_detail, s_detail = {}, {}, {}

		#lane_pred, det_pred, scn_pred = preds

		# Lane estimation loss (only heatmaps)
		if self.lane_det:
			lane_true = targets["lane_det"]
			lane_pred = preds["lane_det"]
			l_loss, l_detail = self.lanes_loss(lane_pred, lane_true)

		# Object detection loss (only heatmaps)
		if self.obj_det:
			det_true = targets["obj_det"]
			det_pred = preds["obj_det"]
			d_loss, d_detail = self.objects_loss(det_pred, det_true)

		# Scene classification loss
		if self.scene_cls:
			scn_true = targets["scn_cls"]
			scn_pred = preds["scn_cls"]
			scn_loss, s_detail = self.cls_loss(scn_pred, scn_true)

		loss_detail = {k : v for d in (d_detail, l_detail, s_detail) for k,v in d.items()}

		return l_loss + d_loss + scn_loss, loss_detail
