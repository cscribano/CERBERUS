# -*- coding: utf-8 -*-
# ---------------------

import torch
from torch import nn

from models.backbones.resnet import *
from models.backbones.efficientnet import *
from models.backbones.mobilenetv2 import *

from models.necks import SimpleNeck, BiFPNNeck
from models.heads import ScnHead, ObjectHead, LaneHead
from utils.decoding import kp_from_heatmap

from conf import Conf

class CerberusModel(nn.Module):
    def __init__(self, cnf):
        # type: (Conf) -> None
        super(CerberusModel, self).__init__()
        self.cnf = cnf

        # Configuration
        self.lane_det = cnf.base.get("lane_det", True)
        self.obj_det = cnf.base.get("object_det", True)
        self.scene_cls = cnf.base.get("scene_cls", True)
        self.obj_occl = cnf.base.get("occlusion_cls", True)

        self.det_classes = cnf.base.get("det_classes", 10)
        self.lane_classes = cnf.base.get("lane_classes", 8)
        scn_classes = cnf.base.get("scn_classes", {})
        self.scn_classes = [v for v in scn_classes.values()]
        
        # Backbone
        assert self.lane_det or self.obj_det, "At least one task must be enabled!"
        self.backbone = eval(cnf.model.backbone.name)(**cnf.model.backbone.args)

        # Neck
        self.neck = eval(cnf.model.neck.name)(self.backbone.outplanes, **cnf.model.neck.args)

        # LANE DETECTION HEAD
        if self.lane_det:
            self.lane_q_offsets = cnf.base.get("lane_q_offsets", False)
            self.head_lane = LaneHead(num_classes=self.lane_classes, in_channels=self.neck.out_channels,
                                      conv_channels=cnf.model.head_channel, quant_offsets=self.lane_q_offsets)

        # OBJECT DETECTION HEAD
        if self.obj_det:
            self.head_obj = ObjectHead(num_classes=self.det_classes,
                                       in_channels=self.neck.out_channels, conv_channels=cnf.model.head_channel)

        # SCENE CLASSIFICATION HEAD
        if self.scene_cls:
            self.head_scn = ScnHead(in_channels=self.neck.out_channels,
                                    classes=self.scn_classes)

    def forward(self, x, inference=False):
        # type: (torch.tensor, bool) -> dict[str, torch.Tensor, ...]

        # Features
        feats = self.backbone(x)

        # Upsample
        big, small = self.neck(feats)

        # Output
        outputs = {}

        if self.lane_det:
            lane_out = self.head_lane(big, nms=inference)
            outputs["lane_det"] = lane_out

        if self.obj_det:
            obj_out = self.head_obj(big, nms=inference)
            outputs["obj_det"] = obj_out

        if self.scene_cls:
            scn_out = self.head_scn(small, argmax=inference)
            outputs["scn_cls"] = scn_out

        return outputs

    def inference(self, x, benchmarking=False):

        assert x.shape[0] == 1, "Only BS=1 is supported!"

        # inference
        predictions = self.forward(x, inference=True)

        if benchmarking:
            return predictions

        # ------------------
        # Lane decoding
        # ------------------
        if self.lane_det:
            lane_preds = predictions["lane_det"]
            hm_lane, ofs_lane, = lane_preds["heatmaps"], lane_preds["offsets"]

            l_scores, l_indices, l_labels = kp_from_heatmap(hm_lane, th=0.6, pseudo_nms=False)
            l_votes = ofs_lane[0, :, l_indices[:, 1], l_indices[:, 0]] * 4

            if self.lane_q_offsets:
                quant_ofs = lane_preds["quant"]
                quant_ofs = quant_ofs[0, :, l_indices[:, 1], l_indices[:, 0]]
                l_indices = l_indices.float()
                l_indices[:, 1] += quant_ofs[1]
                l_indices[:, 0] += quant_ofs[0]

            l_indices = l_indices * 4
            lanes = torch.cat([l_indices.float(), l_scores.unsqueeze(-1)], dim=-1)

            lane_pred = {
                "lanes": lanes,
                "lanes_labels": l_labels,
                "lanes_votes": l_votes
            }

            predictions["lane_det"]["decoded"] = lane_pred

        # ------------------
        # Boxes decoding
        # ------------------
        if self.obj_det:
            det_preds = predictions["obj_det"]
            hm_det, ofs_det, occlu_det = det_preds["heatmaps"], det_preds["offsets"], det_preds["occlusion"]
            d_scores, d_indices, d_labels = kp_from_heatmap(hm_det, th=0.6, pseudo_nms=False)

            bb_ofs = ofs_det[0, :, d_indices[:, 1], d_indices[:, 0]]
            x1x2 = (bb_ofs[:2] + d_indices[..., 0].unsqueeze(0)) * 4
            y1y2 = (bb_ofs[2:] + d_indices[..., 1].unsqueeze(0)) * 4

            # better safe than sorry
            x1x2 = torch.clip(x1x2, 0, 640)
            y1y2 = torch.clip(y1y2, 0, 320)

            boxes = torch.stack([x1x2[0], y1y2[0], x1x2[1], y1y2[1], d_scores], dim=-1)

            det_pred = {
                "boxes": boxes,
                "labels": d_labels
            }

            if self.obj_occl:
                occl = occlu_det[0, 0, d_indices[:, 1], d_indices[:, 0]].sigmoid()
                det_pred["occlusion"] = occl

            predictions["obj_det"]["decoded"] = det_pred

        return predictions

if __name__ == '__main__':
    from torchinfo import summary

    cnf = Conf(exp_name='mobilenetv2_bifpn', log=False)
    model = CerberusModel(cnf).cuda()
    summary(model, input_size=(1, 3, 640,320), depth=5)

    x = torch.rand((1,3,320,640), dtype=torch.float32).cuda()
    y = model(x)
