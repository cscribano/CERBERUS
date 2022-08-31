# -*- coding: utf-8 -*-
# ---------------------
from abc import ABCMeta

from albumentations import Compose, KeypointParams, BboxParams, \
    RandomBrightnessContrast, GaussNoise, RGBShift, CLAHE,\
    RandomGamma, HorizontalFlip, Resize, Normalize, CenterCrop, RandomCrop, ShiftScaleRotate
from albumentations.pytorch.transforms import ToTensorV2

class BaseTransform(object, metaclass=ABCMeta):
    def __init__(self, w, h, input_w, input_h):

        # Find resize dimension (before crop)
        ws = w // input_w
        hs = h // input_h
        s = min(ws, hs)
        self.rw, self.rh = int(w // s), int(h // s)

        self.tsfm = ...

    def __call__(self, img, keypoints=None, kp_labels=None, kp_ids = None, bboxes=None, bb_labels=None, bb_occl=None):
        if keypoints is None:
            keypoints = []
            kp_labels = []
            kp_ids = []
        if bboxes is None:
            bboxes = []
            bb_labels = []
            bb_occl = []

        augmented = self.tsfm(image=img, keypoints=keypoints, kp_labels=kp_labels,
                              kp_ids=kp_ids, bboxes=bboxes, bb_labels=bb_labels, bb_occl=bb_occl)
        img, kp, kp_l, kp_i, bb, bb_l, bb_o = augmented['image'], augmented['keypoints'], augmented['kp_labels'],\
                                  augmented['kp_ids'], augmented['bboxes'], augmented['bb_labels'], augmented['bb_occl']
        return img, kp, kp_l, kp_i, bb, bb_l, bb_o

class RandomAspect(BaseTransform):
    def __init__(self, w, h, input_w, input_h):
        super().__init__(w, h, input_w, input_h)

        self.tsfm = Compose([
            Resize(self.rh, self.rw),
            ShiftScaleRotate(),
            # CenterCrop(320, 640),
            RandomCrop(320, 640),
            HorizontalFlip(),
            RandomBrightnessContrast(0.4, 0.4),
            GaussNoise(),
            RGBShift(),
            CLAHE(),
            RandomGamma(),
            Normalize(),
            ToTensorV2()
        ], keypoint_params=KeypointParams(format='xy', label_fields=['kp_labels', 'kp_ids']),
            bbox_params=BboxParams(format='pascal_voc', label_fields=['bb_labels', 'bb_occl']))


class Preproc(BaseTransform):
    def __init__(self, w, h, input_w, input_h):
        super().__init__(w, h, input_w, input_h)

        self.tsfm = Compose([
            Resize(self.rh, self.rw),
            CenterCrop(320, 640),
            Normalize(),
            ToTensorV2()
        ], keypoint_params=KeypointParams(format='xy', label_fields=['kp_labels', 'kp_ids']),
            bbox_params=BboxParams(format='pascal_voc', label_fields=['bb_labels', 'bb_occl']))

