# -*- coding: utf-8 -*-
# ---------------------

from tqdm import tqdm
from pprint import pprint

import cv2
import click
import torch
import numpy as np

from torchinfo import summary
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import F1Score, Accuracy
from torchmetrics import JaccardIndex
from torchvision import transforms

from conf import Conf
from models import CerberusModel
from utils.box_utils import match_bboxes
from inference.postproc import cluster_lane_preds, fit_lanes
from dataset import MultitaskDataset, ignore_collate


@click.command()
@click.option('--conf_file', '-c', type=click.Path(exists=True), default=None, required=True)
@click.option('--weights_file', '-w', type=click.Path(exists=True), default=None, required=False)
@click.option('--show', '-s', type=click.BOOL, default=False, required=False)
def main(conf_file, weights_file, show):

	cnf = Conf(conf_file_path=conf_file, log=False)
	cnf.dataset.images_root = "/home/carmelo/DATASETS/BDD100K/bdd100k_images/images/100k"
	cnf.dataset.lane_det.data_root = "/home/carmelo/CEMP/MT_ADASNET/data"
	cnf.dataset.obj_det.data_root = "/home/carmelo/DATASETS/BDD100K/bdd100k_det/labels/det_20"

	# Select tasks
	eval_lane_det = cnf.base.get("lane_det", True)
	eval_obj_det = cnf.base.get("object_det", True)
	eval_obj_occl = cnf.base.get("occlusion_cls", True)
	eval_scene_cls = cnf.base.get("scene_cls", True)

	device = "cuda" if torch.cuda.is_available() else 'cpu'

	# Inverse normalization (for display)
	invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
														std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
								   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
														std=[1., 1., 1.]),
								   ])

	# Torchmetrics
	map = MeanAveragePrecision()
	iou = JaccardIndex(num_classes=2)

	wtr_f1 = F1Score(num_classes=7, average='micro')
	scn_f1 = F1Score(num_classes=7, average='micro')
	td_f1 = F1Score(num_classes=4, average='micro')

	occl_acc = Accuracy()

	# Load data
	collate_fn = ignore_collate(["centers", "offsets", "keypoints",
								 "occlusion", "boxes", "classes", "lanes"])

	valset = MultitaskDataset(cnf, mode="val", gt=True)
	val_loader = DataLoader(valset, collate_fn=collate_fn, batch_size=1)

	# load model
	model = CerberusModel(cnf).to(device)
	ck = torch.load(weights_file, map_location=device)
	model.load_state_dict(ck, strict=True)

	model.eval()

	# Print stats
	# summary(model, input_size=(1, 3, 640, 320))

	# Run evaluation loop
	for batch_idx, batch in enumerate(tqdm(val_loader)):
		img, targets = batch
		img = img.to(cnf.device)

		with torch.no_grad():
			pred = model.inference(img)

		"""det_out, lane_out, scn_out, heatmaps_out = pred
		boxes, boxes_cls, boxes_occl = 
		lanes, lanes_cls, lanes_votes = lane_out"""

		# =======================
		# Object detection metric
		# =======================
		if eval_obj_det:
			det_out = pred["obj_det"]["decoded"]
			boxes, boxes_cls = det_out["boxes"], det_out["labels"]

			car_pred = torch.nonzero(boxes_cls == 2).squeeze(1)
			det_pred = {
				'boxes': boxes[:, :4].cpu(),
				'scores': boxes[:, 4].cpu(),
				'labels': boxes_cls.cpu(),
			}

			car_true = (targets["obj_det"]["classes"][0] == 2).nonzero().squeeze(1)
			det_target = {
				'boxes': targets["obj_det"]["boxes"][0],
				'labels': targets["obj_det"]["classes"][0],
			}

			if eval_obj_occl:
				boxes_occl = det_out["occlusion"]
				det_pred['occlusion'] =  boxes_occl.cpu()
				det_target['occlusion'] = targets["obj_det"]["occlusion"][0]

			# TODO: testare con decodifica del GT!
			map.update([det_pred], [det_target])

			# -------------------------------
			# Occlusion Classification Metric
			# -------------------------------
			if eval_obj_occl:
				gt_valid, pred_valid, _, _ = match_bboxes(det_target["boxes"][car_true], det_pred["boxes"][car_pred])
				occlu_true = det_target["occlusion"][car_true][gt_valid].int()
				occlu_pred = det_pred["occlusion"][car_pred][pred_valid]

				if len(gt_valid) >= 1:
					occl_acc.update(occlu_pred, occlu_true)

		# ===============================
		# Scene Classification Metric
		# ===============================
		if eval_scene_cls:
			scn_out = pred["scene_cls"]
			wtr_f1.update(scn_out['weather'].cpu(), targets['scn_cls']['weather'])
			scn_f1.update(scn_out['scene'].cpu(), targets['scn_cls']['scene'])
			td_f1.update(scn_out['timeofday'].cpu(), targets['scn_cls']['timeofday'])

		# =======================
		# Lane Estimation Metric
		# =======================
		if eval_lane_det:
			lane_out = pred["lane_est"]["decoded"]
			lanes, lanes_cls, lanes_votes = lane_out["lanes"], lane_out["lanes_labels"], lane_out["lanes_votes"]

			# Build GT mask
			gt_lanes = targets["lane_det"]["lanes"][0]
			gt_lanes = [l.numpy() * 4 for l in gt_lanes]
			gt_mask = lanes_to_mask(gt_lanes, cnf.dataset.input_h, cnf.dataset.input_w)
			gm = torch.from_numpy(gt_mask).long().unsqueeze(0)

			# Build predicted mask
			lane_clusters = cluster_lane_preds(lanes, lanes_cls, lanes_votes)
			lanes_pred = fit_lanes(lane_clusters)

			pred_lanes = []
			for i in range(8):
				pred_lanes += lanes_pred[i]

			pred_mask = lanes_to_mask(pred_lanes, cnf.dataset.input_h, cnf.dataset.input_w)
			pm = torch.from_numpy(pred_mask).long().unsqueeze(0)

			iou.update(gm, pm)

		#if batch_idx > 500:
		#	break

		# Display results
		if show:
			frame = invTrans(img[0])
			frame = frame.cpu().numpy().transpose(1, 2, 0)

			if eval_obj_det:
				# true
				boxes_pred = boxes[:, :4].cpu().numpy()
				for b in boxes_pred:
					color = (0, 255, 0)
					frame = cv2.rectangle(frame, (int(b[2]), int(b[3])), (int(b[0]), int(b[1])), color, 2)

				# objects pred
				boxes_true = targets["obj_det"]["boxes"][0]
				for b in boxes_true:
					color = (0, 0, 255)
					frame = cv2.rectangle(frame, (int(b[2]), int(b[3])), (int(b[0]), int(b[1])), color, 2)

			#Lane masks
			if eval_lane_det:
				all_mask = np.zeros((cnf.dataset.input_h, cnf.dataset.input_w, 3), dtype=np.uint8)
				all_mask[:, :, 1] = pred_mask*255
				all_mask[:, :, 2] = gt_mask*255

			while cv2.waitKey(1) != ord('q'):
				if eval_obj_det: cv2.imshow("detection", frame)
				if eval_lane_det: cv2.imshow("lanes", all_mask)



	if eval_obj_det:
		print("--- OBJECT DETECTION ---")
		pprint(map.compute())

	if eval_lane_det:
		print("--- LANE ESTIMATION ---")
		pprint(iou.compute())

	if eval_scene_cls:
		print("--- SCENE CLASSIFICATION F1 (weather, scene, time of day) ---")
		pprint(wtr_f1.compute())
		pprint(scn_f1.compute())
		pprint(td_f1.compute())

	if eval_obj_det and eval_obj_occl:
		print("--- OCCLUSION CLASSIFICATION ACCURACY ---")
		pprint(occl_acc.compute())

def lanes_to_mask(lanes, h, w):
	gt_mask = np.zeros((h, w), dtype=np.uint8)
	for l in lanes:
		points = l.astype(np.int32)

		# Draw mask
		points = points.reshape((-1, 1, 2))
		gt_mask = cv2.polylines(gt_mask, [points], False, (1), 2)

	return gt_mask

if __name__ == '__main__':
	# baseline:  'map_50': tensor(0.5604),
	main()

