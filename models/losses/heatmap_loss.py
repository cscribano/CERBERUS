# -*- coding: utf-8 -*-
# ---------------------

import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
	'CornerNetFocalLoss', 'QualityFocalLoss', 'AdaptiveWingLoss', 'WMSELoss'
]

# reference: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/gaussian_focal_loss.py
# https://github.com/gau-nernst/centernet-lightning/blob/9fa4571904f1d68703f1cf4fa6e93e3c53d2971f/centernet_lightning/losses/heatmap_losses.py
class CornerNetFocalLoss(nn.Module):
	"""CornerNet Focal Loss. Use logits to improve numerical stability. CornerNet: https://arxiv.org/abs/1808.01244
	"""

	# reference implementations
	# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/gaussian_focal_loss.py
	def __init__(self, alpha: float = 2, beta: float = 2, reduction: str = "mean"):
		"""CornerNet Focal Loss. Default values from the paper

		Args:
			alpha: control the modulating factor to reduce the impact of easy examples. This is gamma in the original Focal loss
			beta: control the additional weight for negative examples when y is between 0 and 1
			reduction: either none, sum, or mean
		"""
		super().__init__()
		assert reduction in ("none", "sum", "mean")
		self.alpha = alpha
		self.beta = beta
		self.reduction = reduction

	def forward(self, inputs: torch.Tensor, targets: torch.Tensor):

		pos_inds = targets.eq(1).float()
		neg_inds = targets.lt(1).float()

		neg_weights = torch.pow(1 - targets, 4)
		# clamp min value is set to 1e-12 to maintain the numerical stability
		pred = torch.clamp(inputs, 1e-12)

		pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
		neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

		num_pos = pos_inds.float().sum()
		pos_loss = pos_loss.sum()
		neg_loss = neg_loss.sum()

		if num_pos == 0:
			loss = -neg_loss
		else:
			loss = -(pos_loss + neg_loss) / num_pos

		return loss

class WMSELoss(nn.Module):

	def __init__(self, alpha: float=4, beta: float = 2, reduction: str = 'mean'):

		super().__init__()
		assert reduction in ('none', 'sum', 'mean')
		self.alpha = alpha
		self.beta = beta
		self.reduction = reduction

	def forward(self, inputs: torch.Tensor, targets: torch.Tensor):

		mse = F.mse_loss(inputs, targets, reduction='none')
		mf_t = (torch.pow(1 + targets, self.alpha))
		mf_p = (torch.pow(1 + inputs.detach(), self.beta))
		modulating_factor = torch.maximum(mf_t, mf_p)
		# modulating_factor = torch.pow(1 + torch.abs(targets.detach() - inputs), self.beta)

		loss = modulating_factor * mse
		if self.reduction == 'none':
			return loss

		bs = loss.shape[0]
		loss = torch.sum(loss)
		if self.reduction == 'mean':
			loss = loss / (1 + targets.gt(0.96).sum().float())
			loss = loss / bs

		return loss


class QualityFocalLoss(nn.Module):
	"""Quality Focal Loss. Generalized Focal Loss: https://arxiv.org/abs/2006.04388
	"""

	def __init__(self, beta: float = 2, reduction: str = "mean"):
		"""Quality Focal Loss. Default values are from the paper

		Args:
			beta: control the scaling/modulating factor to reduce the impact of easy examples
			reduction: either none, sum, or mean
		"""
		super().__init__()
		assert reduction in ("none", "sum", "mean")
		self.beta = beta
		self.reduction = reduction

	def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
		probs = torch.sigmoid(inputs)

		ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
		modulating_factor = torch.abs(targets - probs) ** self.beta

		loss = modulating_factor * ce_loss

		if self.reduction == "sum":
			return torch.sum(loss)

		if self.reduction == "mean":
			return torch.sum(loss) / targets.eq(1).float().sum()

		return loss

# torch.log  and math.log is e based
# https://github.com/elliottzheng/AdaptiveWingLoss/blob/master/adaptive_wing_loss.py
class AdaptiveWingLoss(nn.Module):
	def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
		super(AdaptiveWingLoss, self).__init__()
		self.omega = omega
		self.theta = theta
		self.epsilon = epsilon
		self.alpha = alpha

	def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
		'''
		:param pred: BxNxHxH
		:param target: BxNxHxH
		:return:
		'''

		y = targets
		y_hat = inputs
		delta_y = (y - y_hat).abs()
		delta_y1 = delta_y[delta_y < self.theta]
		delta_y2 = delta_y[delta_y >= self.theta]
		y1 = y[delta_y < self.theta]
		y2 = y[delta_y >= self.theta]
		loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
		A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
			torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
		C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
		loss2 = A * delta_y2 - C
		return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
