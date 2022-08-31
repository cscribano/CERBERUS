# -*- coding: utf-8 -*-
# ---------------------

import cv2
import click
from pathlib import Path

from math import ceil
import numpy as np
import torch

from heatmap_utils import kps_to_heatmaps

IMG_WIDTH = 1280
IMG_HEIGHT = 720

I_SCALE = 2
O_SCALE = 4

PPK = 25  # Number of pixels per keypoint

CLS = {
	'single yellow': 0,
	'single white': 1,
	'crosswalk': 2,
	'double white': 3,
	'double other': 4,
	'road curb': 5,
	'single other': 6,
	'double yellow': 7
}

@click.command()
@click.option('--img_root', '-i', type=click.Path(exists=True), default=None, required=False)
def main(img_root):
	# type: (Path) -> None

	split = "val"

	# Load Images
	img_root = Path(img_root) / split
	images = {p.name: p for p in img_root.glob("*.jpg")}
	# Load annotation file
	annot_file = Path(f"{split}_{PPK}.pt")
	annotations = torch.load(annot_file)

	# List lane classes
	classes = set([l['category'] for a in annotations for l in a.get('labels', [])])
	print(classes)

	w = ceil(IMG_WIDTH / O_SCALE)
	h = ceil(IMG_HEIGHT / O_SCALE)


	for lanes in annotations:
		lbc = [[] for _ in range(8)]

		labels = lanes.get("labels", [])
		for l in labels:
			cls_id = CLS[l['category']]
			lbc[cls_id] += l["keypoints"]

		# Load image
		image_file = img_root / images[lanes["name"]]
		frame = cv2.imread(str(image_file))

		# Numpy
		lane_np = np.concatenate([np.array(l['keypoints']) for l in labels]).astype(np.int32)
		for c in lane_np:
			frame = cv2.circle(frame, (c[0], c[1]), 3, (0, 255, 0), thickness=3)

		# Generate heatmaps
		n = torch.tensor([IMG_WIDTH, IMG_HEIGHT])
		lbc = [torch.tensor(l) / n if len(l) > 0 else torch.tensor(l) for l in lbc]
		heatmaps = kps_to_heatmaps(lbc, w, h, sigma=2)

		# Display
		hm_show, _ = torch.max(heatmaps, dim=0)
		hm_show = hm_show.numpy() * 255
		hm_show = hm_show.astype(np.uint8)
		hm_show = cv2.applyColorMap(hm_show, cv2.COLORMAP_JET)



		cv2.imshow("heatmap", hm_show)
		cv2.imshow("frame", frame)
		while cv2.waitKey(1) != ord('q'):
			pass


if __name__ == '__main__':
	main()
