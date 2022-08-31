# -*- coding: utf-8 -*-
# ---------------------

from conf import Conf

import torch
import numpy as np
import cv2

import pytorch_lightning as pl
import torchvision as tv
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers

from torch.optim import *
from utils.lr_scheduler import *

from models import CerberusModel
from models.losses import MultiTaskLoss
from dataset import MultitaskDataset, ignore_collate


class PL_trainable(pl.LightningModule):
    def __init__(self, cnf):
        super().__init__()

        self.cnf = cnf
        self.backbone = CerberusModel(cnf)
        self.criterion = MultiTaskLoss(cnf)

        self.plot_images = 10

    def forward(self, img):
        pred = self.backbone(img)
        return pred

    def training_step(self, batch, batch_idx):

        img, targets = batch
        preds = self.forward(img)

        # Loss
        loss, loss_detail = self.criterion(preds, targets)

        # single scheduler
        sch = self.lr_schedulers()
        sch.step()

        lr = sch.get_last_lr()[0]
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        self.log('lr', lr, on_step=True, on_epoch=False)
        for k, v in loss_detail.items():
            self.log(f'train_{k}_loss', v, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        # Inference
        img, targets = batch
        preds = self.forward(img)

        # Loss
        loss, loss_detail = self.criterion(preds, targets)

        # plot
        if self.plot_images > 0:
            true, pred = [], []

            if self.cnf.base.get("object_det", False):
                true.append(targets["obj_det"]["heatmaps"])
                pred.append(preds["obj_det"]["heatmaps"])

            if self.cnf.base.get("lane_det", False):
                true.append(targets["lane_det"]["heatmaps"])
                pred.append(preds["lane_det"]["heatmaps"])

            true = torch.cat(true, dim=1)
            pred = torch.cat(pred, dim=1)
            img_resize = torch_input_img(img[0].cpu().detach())
            hm_true = torch_heatmap_img(true[0].cpu().detach())
            hm_pred = torch_heatmap_img(pred[0].cpu().detach())
            grid = torch.stack([img_resize, hm_true, hm_pred], dim=0)

            grid = tv.utils.make_grid(grid.float())
            self.logger.experiment.add_image(tag=f'results_{self.plot_images}',
                                             img_tensor=grid, global_step=self.global_step)
            self.plot_images -= 1

        # Log
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        for k, v in loss_detail.items():
            self.log(f'val_{k}_loss', v, on_step=False, on_epoch=True)

        return loss


    def test_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs) -> None:
        self.plot_images = 10

    def configure_optimizers(self):
        optimizer = eval(self.cnf.optimizer.name)(self.parameters(), **self.cnf.optimizer.args)

        if self.cnf.lr_scheduler.get("name", None) is not None:
            scheduler = eval(self.cnf.lr_scheduler.name)(optimizer, **self.cnf.lr_scheduler.args)
            return [optimizer], [scheduler]
        return [optimizer]

    def on_validation_epoch_end(self):
        torch.save(self.backbone.state_dict(), f'{self.cnf.exp_log_path}/last.pth')


def torch_heatmap_img(heatmap):
    hm_show, _ = torch.max(heatmap, dim=0)
    hm_show = hm_show.numpy() * 255
    hm_show = hm_show.astype(np.uint8)
    hm_show = cv2.applyColorMap(hm_show, cv2.COLORMAP_JET)
    hm_show = cv2.cvtColor(hm_show, cv2.COLOR_BGR2RGB)
    hm_show = cv2.resize(hm_show, (640, 480)) / 255

    return torch.from_numpy(hm_show).permute(2, 0, 1)


def torch_input_img(img):
    invTrans = tv.transforms.Compose([tv.transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   tv.transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    img = invTrans(img)
    img = tv.transforms.Resize((480, 640))(img)
    return img

def trainer_run(cnf):
    # type: (Conf) -> None

    # ------------
    # data
    # ------------
    trainset = MultitaskDataset(cnf, mode="train")
    valset = MultitaskDataset(cnf, mode="val")

    collate_fn = ignore_collate(["centers", "offsets", "keypoints", "occlusion", "quant_offsets"])
    train_loader = DataLoader(trainset, collate_fn=collate_fn, **cnf.dataset.train_dataset.loader_args)
    val_loader = DataLoader(valset, collate_fn=collate_fn, **cnf.dataset.val_dataset.loader_args)

    # ------------
    # model
    # ------------
    model = PL_trainable(cnf)

    # ------------
    # training
    # ------------
    gpus = [0]
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cnf.exp_log_path, name="", version="")

    trainer = pl.Trainer(default_root_dir=cnf.exp_log_path, logger=tb_logger,
                         max_epochs=cnf.epochs, gpus=gpus)
    trainer.fit(model, train_loader, val_loader)

