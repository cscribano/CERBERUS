# -*- coding: utf-8 -*-
# ---------------------

import os
import cv2
import click
import json
from pathlib import Path

import torch
from tqdm import tqdm

import numpy as np
import cv2

from tools import *

IMG_WIDTH = 1280
IMG_HEIGHT = 720

I_SCALE = 2
O_SCALE = 8

PPK = 25 # Number of pixels per keypoint


@click.command()
@click.option('--img_root', '-i', type=click.Path(exists=True), default=None, required=False)
@click.option('--labels_root', '-l', type=click.Path(exists=True), default=None, required=True)
@click.option('--display', '-d', type=bool, default=True, required=False)
@click.option('--out', '-o', type=click.Path(), default=Path('.'), required=True)
def main(img_root, labels_root, display, out):
	# type: (Path, Path, bool, Path) -> None

	if display:
		assert img_root is not None

	img_root = Path(img_root)
	labels_root = Path(labels_root)

	for split in ["val"]: #, "train"]:

		print(f"=> Processing split {split}")

		out_file = out / f"{split}_{PPK}_new.pt"

		# Load annotation file
		masks_path = labels_root / "masks"/ split

		polygons_file = labels_root / "polygons" / f"lane_{split}.json"
		polygons = json.load(open(polygons_file, "r"))

		for p_index, p_lane in enumerate(tqdm(polygons)):

			frame = None
			lanes = None

			# Broken annotation, skip....
			if type(p_lane) == list:
				continue

			if display:

				# Load frame
				image_file = p_lane['name']
				image_file = img_root / split / image_file
				frame = cv2.imread(str(image_file))

				# Load mask
				mask_file = Path(p_lane['name']).stem + '.png'
				mask_file = masks_path / mask_file
				mask = cv2.imread(str(mask_file))#[..., 0]

				"""
				for p in mask[mask != 255]:
					d = (p & 32) >> 5  # direction (parallel or perpendicular)
					s = (p & 16) >> 4  # style (full or dashed)
					b = (p & 8) >> 3  # background (lane (0) or background (1))
					c = (p & 7)  # class (road curb, crosswalk, double white, double yellow, double other color,
				# single white, single yellow, single other color.) (8)
				"""
				lanes = 1-((mask & 8) >> 3)  # direction (parallel or perpendicular)

			labels = p_lane.get('labels', None)
			if labels is None:
				continue

			for il, l in enumerate(labels):
				assert len(l["poly2d"]) == 1
				pts = l["poly2d"][0]['vertices']

				# Define number of points according to length
				nppt = np.array(pts)
				tot_l = 0
				for ip, p in enumerate(nppt):
					if ip == len(pts) - 1:
						break

					l = np.linalg.norm(p-nppt[ip+1])
					tot_l += l

				# Compute beizer cube curve
				xvals, yvals = bezier_curve(pts, nTimes=max(3, int(tot_l//PPK)))
				pt = np.stack([xvals, yvals], axis=-1) #.astype(np.int32)
				labels[il]['keypoints'] = pt#.tolist()

			# ---- Filter double lines ----
			all_dist = []
			for i1, l1 in enumerate(labels):

				i_dist = []
				for i2, l2 in enumerate(labels):
					d = dist(l1, l2)
					i_dist.append(d)

				all_dist.append(i_dist)

			all_dist = np.array(all_dist)
			min_dist = np.argmin(all_dist, -1)

			pairs = []
			for id, d in enumerate(min_dist):
				if min_dist[id] == d and min_dist[d] == id:
					if [d, id] not in pairs:
						pairs.append([id, d])

			# Replace double lines with mean line
			for p in pairs:
				if all_dist[p[0], p[1]] < 80:

					# Compute mean line
					l1 = labels[p[0]]
					l2 = labels[p[1]]
					pt1, pt2 = compare_labels(l1, l2)
					pt3 = (pt1 + pt2) / 2

					# Fit new curve
					n = pt3.shape[0]
					x, y = pt3[:, 0], pt3[:, 1]

					if n > 3:
						v = get_bezier_parameters(x, y)
						xvals, yvals = bezier_curve(v, nTimes=n)
						pt3 = np.stack([xvals, yvals], axis=-1)

					# Update
					labels[p[0]]["keypoints"] = pt3
					labels[p[1]] = None

			# plot
			if display:
				for l in labels:
					if l is None:
						continue

					pt = l["keypoints"]
					pt = np.array(pt).astype(np.int32)
					for c in pt:
						frame = cv2.circle(frame, (c[0], c[1]), 3, (0,255,0), thickness=3)

			# Append
			polygons[p_index]['labels'] = labels

			# Display result
			if display:
				cv2.imshow("frame", frame)
				cv2.imshow("lanes", lanes * 255)

				while cv2.waitKey(1) != ord('q'):
					pass

		# Save
		#torch.save(polygons, out_file)


if __name__ == '__main__':
	main()
	"""
	-i /home/carmelo/DATASETS/BDD100K/bdd100k_images/images/100k
	-l /home/carmelo/DATASETS/BDD100K/bdd100k_lanes/labels/lane
	"""
