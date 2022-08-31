# -*- coding: utf-8 -*-
# ---------------------

from pathlib import Path

import cv2
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.utils import LaneProcessor, DetProcessor
from dataset.utils.heatmaps import CornerNetRadius, FixedRadius
from dataset.utils.transforms import RandomAspect, Preproc

from conf import Conf

class MultitaskDataset(Dataset):

    WIDTH = 1280
    HEIGHT = 720

    def __init__(self, cnf, mode='train', gt=False):
        # type: (Conf,str, bool) -> None
        """
        :param cnf: Configuration file
        :param mode: mode string (only train supported)
        :param gt: return GT data for test
        """

        self.cnf = cnf
        self.mode = mode
        self.return_gt = gt
        assert mode in ['train', 'val']

        # Image and heatmap sizes
        self.det_classes = cnf.base.get("det_classes", 10)
        self.lane_classes = cnf.base.get("lane_classes", 8)

        self.output_s = cnf.dataset.output_stride
        self.input_w, self.input_h = cnf.dataset.input_w, cnf.dataset.input_h
        self.target_w, self.target_h = int(self.input_w // self.output_s), int(self.input_h // self.output_s)

        # transforms
        self.transforms = RandomAspect(self.WIDTH, self.HEIGHT, self.input_w, self.input_h) \
            if mode in ["train", "train_all"] else Preproc(self.WIDTH, self.HEIGHT, self.input_w, self.input_h)

        # Image files
        self.images_root = Path(cnf.dataset.images_root) / mode
        self.image_files = {p.name: p for p in self.images_root.rglob("*.jpg")}

        # Lane keypoints
        self.lane_det = cnf.base.get("lane_det", False)
        if self.lane_det:
            lane_annot_file = Path(cnf.dataset.lane_det.data_root) / f"{mode}_{cnf.dataset.lane_det.ppm}_new.pt"
            lane_annotations = torch.load(lane_annot_file)
            self.lane_annotations = {v["name"]: v for v in lane_annotations}
        else:
            lane_annotations = []
            self.lane_annotations = {}

        # Object detection annotations
        self.obj_det = cnf.base.get("object_det", False)
        if self.obj_det:
            det_annot_file = Path(cnf.dataset.obj_det.data_root) / f"det_{mode}.json"
            det_annotations = json.load(open(det_annot_file, "r"))
            self.det_annotations = {v["name"]: v for v in det_annotations}

            # Heatmaps configuration
            s = cnf.dataset.obj_det.get("sigma", None)
            if s is not None:
                self.det_sigma = eval(s.name)(**s.args)
            else:
                self.det_sigma = FixedRadius(r=2)

        else:
            det_annotations = []
            self.det_annotations = {}
            self.det_sigma = lambda x, y: 1 # Identity

        # Target generators
        self.lane_processor = LaneProcessor(self.lane_classes, self.output_s, self.target_w, self.target_h)
        self.det_processor = DetProcessor(self.det_classes, self.output_s, self.target_w, self.target_h)

        # Multi-Task worthy dataset elements
        if self.obj_det and self.lane_det:
            # Intersection
            self.annot_keys = set([v['name'] for v in det_annotations]) & \
                               set([v['name'] for v in lane_annotations])
            self.annot_keys = list(self.annot_keys)

        else:
            # Concatenation (one list will be empty)
            self.annot_keys = [v['name'] for v in det_annotations] +\
                               [v['name'] for v in lane_annotations]

        self.annot_keys.sort()  # For Validation reproducibility

        assert len(self.annot_keys) > 0

    def __len__(self):
        # type: () -> int
        return len(self.annot_keys)

    def __getitem__(self, i):
        # type: (int) -> tuple[torch.tensor, ...]

        target = {}

        # Select annotation
        annot_name = self.annot_keys[i]

        # Load image
        img_file = self.image_files[annot_name]
        image = cv2.imread(str(img_file))

        # Retrieve LANE keypoints
        lane_annot = self.lane_annotations.get(annot_name, {})
        lane_labels, lane_kp, lane_cls, lane_ids = self.lane_processor.keypoints(lane_annot)

        # Retrieve OBJECTS boxes
        det_annot = self.det_annotations.get(annot_name, {})
        det_labels, det_bbs, det_cls, occl_cls = self.det_processor.bounding_boxes(det_annot)
        scene_cls = self.det_processor.scene_classification(det_annot)
        target["scn_cls"] = scene_cls

        # Apply transforms
        image, lane_kp, lane_cls, lane_ids, det_bb, det_cls, occl_cls = self.transforms(image, keypoints=lane_kp, kp_labels=lane_cls,
                                                                    kp_ids=lane_ids, bboxes=det_bbs, bb_labels=det_cls, bb_occl=occl_cls)

        # Generate Object detection Target
        if self.obj_det:
            heatmap_det, centers, offsets = self.det_processor.targets(det_labels, det_bb, det_cls)
            occl_cls = torch.tensor(occl_cls).float()

            target["obj_det"] = {
                "heatmaps": heatmap_det,
                "centers": centers,
                "offsets": offsets,
                "occlusion": occl_cls
            }

            if self.return_gt:
                target["obj_det"]["boxes"] = torch.tensor(det_bb)
                target["obj_det"]["classes"] = torch.tensor(det_cls)

                ofstrue = torch.zeros(4, self.target_h, self.target_w)
                ofstrue[:, centers[:, 1].long(), centers[:, 0].long()] = offsets.t().float()
                target["obj_det"]["ofstrue"] = ofstrue

        if self.lane_det:
            # Target heatmap
            heatmap_lane, l_centers, l_offsets,\
                quant_offsets, l_keypoints = self.lane_processor.targets(lane_labels, lane_kp, lane_cls, lane_ids)

            target["lane_det"] = {
                "heatmaps": heatmap_lane,
                "keypoints": l_centers,
                "offsets": l_offsets,
                "quant_offsets": quant_offsets
            }

            if self.return_gt:
                target["lane_det"]["classes"] = torch.tensor(lane_cls)
                target["lane_det"]["lanes"] = l_keypoints

        return image, target

if __name__ == '__main__':

    from tqdm import tqdm
    import numpy as np
    from torch.utils.data import DataLoader

    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])

    cnf = Conf(exp_name='resnet34_bifpn', log=False)
    ds = MultitaskDataset(cnf, **cnf.dataset.train_dataset.args)
    l = len(ds)

    for frame, target in ds:

        hm_det = target["obj_det"]["heatmaps"]
        hm_lane = target["lane_det"]["heatmaps"]

        frame = invTrans(frame)
        frame = frame.numpy().transpose(1, 2, 0)

        hm = torch.cat([hm_det, hm_lane], dim=0)
        hm_show, _ = torch.max(hm, dim=0)
        hm_show = hm_show.numpy() * 255
        hm_show = hm_show.astype(np.uint8)
        hm_show = cv2.applyColorMap(hm_show, cv2.COLORMAP_JET)
        hm_show = cv2.resize(hm_show, (cnf.dataset.input_w, cnf.dataset.input_h))

        super_imposed_img = cv2.addWeighted(hm_show.astype(np.float32) / 255, 0.5, frame, 0.5, 0)

        while cv2.waitKey(1) != ord('q'):
            cv2.imshow("heatmap", hm_show)
            cv2.imshow("frame", super_imposed_img)
