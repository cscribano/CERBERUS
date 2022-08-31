# -*- coding: utf-8 -*-
# ---------------------

import sys
import click
import torch
from pathlib import Path
import logging
from random import randint

import cv2
from torchvision import transforms
from torchinfo import summary

import torch.onnx
import onnx
import onnxsim

from dataset.utils.transforms import Preproc
from dataset.utils.cls import WTR_CLS, SN_CLS, TD_CLS
from models import CerberusModel
from conf import Conf
from inference.postproc import *

from profiler import Profiler

logging.basicConfig(format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

Profiler.set_warmup(10)

cls_col = [(153, 255, 102), (255,255,255), (0, 255, 255), (52, 255, 52), (51, 153, 51),
           (0,255,0), (153,51,51), (0,0,255), (255, 0, 0)]

lancol = [(0,255,255), (255,255,255), (255, 150, 50),(0,0,255),
          (102, 0, 102), (10, 255, 0), (255, 255, 0), (0, 153, 255)]

@click.command()
@click.option('--conf_file', '-c', type=click.Path(exists=True), default=None, required=True)
@click.option('--weights_file', '-w', type=click.Path(exists=True), default=None, required=False)
@click.option('--video', '-v', type=click.Path(exists=True), default=None, required=True)
@click.option('--onnx_export', '-o', type=click.BOOL, default=False, required=False)
@click.option('--max_frames', '-f', type=int, default=None, required=False)
def main(conf_file, weights_file, video, onnx_export, max_frames):

	cnf = Conf(conf_file_path=conf_file, log=False)
	device = "cuda" if torch.cuda.is_available() else 'cpu'

	# load video
	# video = "../videos/dashcam_demo.mp4"
	cap = cv2.VideoCapture(video)
	# cap.set(cv2.CAP_PROP_POS_FRAMES, 150*30)

	# Classes
	wtr = {v: k for k, v in WTR_CLS.items()}
	scn = {v: k for k, v in SN_CLS.items()}
	td = {v: k for k, v in TD_CLS.items()}

	# writer
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	writer = cv2.VideoWriter('../videos/result.mp4', fourcc, 35, (cnf.dataset.input_w*2, cnf.dataset.input_h+40))

	# load model
	model = CerberusModel(cnf).to(device)

	if weights_file is not None:
		ck = torch.load(weights_file, map_location=device)
		model.load_state_dict(ck, strict=True)
	else:
		logging.debug("Weights file not exists!")

	model.eval()

	# Print stats
	summary(model, input_size=(1, 3, 640, 320))

	if onnx_export:
		# Convert to onnx
		print("Converting model to ONNX...")
		dummy_input = torch.randn(1, 3, cnf.dataset.input_h, cnf.dataset.input_w, requires_grad=True)
		dummy_input = dummy_input.to(device)

		base_file = Path(conf_file).stem
		out_file = Path(__file__).parent.parent / 'weights' / f'{base_file}.onnx'
		torch.onnx.export(model, (dummy_input, True), out_file,
		                  input_names=["x"], opset_version=11) # dynamic_axes= {"x": {0: "bs"}},

		# Simplify
		print("Simplifing...")
		model_opt = onnxsim.simplify(str(out_file), skip_fuse_bn=True, dynamic_input_shape=True)
		out_file_sim = Path(__file__).parent.parent / 'weights' / f'{base_file}_sim.onnx'
		onnx.save(model_opt[0], str(out_file_sim))
		print("onnx model simplify Ok!")

	# Image preproc
	pp = Preproc(1280, 720, cnf.dataset.input_w, cnf.dataset.input_h)

	invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
	                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
	                               transforms.Normalize(mean=[-0.485, -0.456, -0.406],
	                                                    std=[1., 1., 1.]),
	                               ])

	frames = 0
	while cap.isOpened():
		with Profiler('acquire'):
			_, frame = cap.read()
			frame_pt = pp(frame)[0]
			frame_pt = frame_pt.unsqueeze(0)
			frame_pt = frame_pt.to(device)

		# Inference
		with torch.no_grad():
			with Profiler('inference_all'):
				pred = model.inference(frame_pt, benchmarking=max_frames is not None)
				if max_frames is not None:
					torch.cuda.synchronize(0)

		if max_frames is None:

			det_out = pred.get("obj_det", None)
			lane_out = pred.get("lane_det", None)
			scn_out = pred.get("scn_cls", None)

			# TODO: exlude disablee tasks
			lane_dec = lane_out["decoded"]
			lanes, lanes_cls, lanes_votes = lane_dec["lanes"], lane_dec["lanes_labels"], lane_dec["lanes_votes"]

			det_dec = det_out["decoded"]
			boxes, boxes_cls, boxes_occl = det_dec["boxes"], det_dec["labels"], det_dec["occlusion"]

			# Show result
			frame = frame_pt[0].cpu()
			frame = invTrans(frame)
			frame = frame.numpy().transpose(1, 2, 0)

			# Classification results
			if scn_out is not None:
				w_cls = wtr[scn_out['weather'].item()]
				s_cls = scn[scn_out['scene'].item()]
				td_cls = td[scn_out['timeofday'].item()]
			else:
				w_cls, s_cls, td_cls = "<UNKNOWN>", "<UNKNOWN>", "<UNKNOWN>"

			with Profiler('lane_clustering'):
				# Lane clustering
				lane_clusters = cluster_lane_preds(lanes, lanes_cls, lanes_votes)

			# Superimpose
			hm_lane, hm_det = lane_out["heatmaps"], det_out["heatmaps"]
			heatmap = torch.cat([hm_lane, hm_det], dim=1)

			hm = heatmap[0].cpu().detach()
			hm_show, _ = torch.max(hm, dim=0)
			hm_show = hm_show.numpy() * 255
			hm_show = hm_show.astype(np.uint8)
			hm_show = cv2.applyColorMap(hm_show, cv2.COLORMAP_JET)
			hm_show = cv2.resize(hm_show, (cnf.dataset.input_w, cnf.dataset.input_h), cv2.INTER_LINEAR)

			super_imposed_img = cv2.addWeighted(hm_show.astype(np.float32) / 255, 0.5, frame, 0.5, 0)

			with Profiler('lane_drawing'):
				# Draw keypoints
				frame = (frame*255).astype(np.uint8)
				lanes_pred = fit_lanes(lane_clusters)

				for lane_cls in range(len(lane_clusters)):
					lanes_pred_cls = lanes_pred[lane_cls]
					col = cls_col[lane_cls]

					for lane_pred in lanes_pred_cls:
						x_new = lane_pred[:, 0]
						y_new = lane_pred[:, 1]

						for cx, cy in zip(x_new, y_new):
							frame = cv2.circle(frame, (int(cx), int(cy)), 1, col, thickness=2, )

			with Profiler('det_drawing'):

				# Draw boxes
				for b, bo in zip(boxes, boxes_occl):
					color = (0, 255, 0) if bo < 0.5 else (0,0,255)
					frame = cv2.rectangle(frame, (int(b[2]), int(b[3])), (int(b[0]), int(b[1])), color, 2)

				write_img = (super_imposed_img * 255).astype(np.uint8)
				write_img = np.concatenate([write_img, frame], axis=1)

			with Profiler('cls_drawing'):
				show_img = np.zeros((cnf.dataset.input_h + 40, cnf.dataset.input_w*2, 3), dtype=np.uint8)
				text = f"WEATHER: {w_cls}   SCENE: {s_cls}   DAYTIME: {td_cls}"
				show_img = cv2.putText(show_img, text, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8,
				                    (255,255,255), 1, cv2.LINE_AA, False)
				show_img[40:, :, :] = write_img

			writer.write(show_img)
			cv2.imshow("result", show_img)

		frames += 1
		k = cv2.waitKey(1)
		if k == ord('q') or (frames == max_frames):
			LOGGER.debug('=================Timing Stats=================')
			LOGGER.debug(f"{'Frame Acquiring:':<37}{Profiler.get_avg_millis('acquire'):>6.3f} ms")
			LOGGER.debug(f"{'Inference total:':<37}{Profiler.get_avg_millis('inference_all'):>6.3f} ms")
			LOGGER.debug(f"{'Lane Clustering:':<37}{Profiler.get_avg_millis('lane_clustering'):>6.3f} ms")
			LOGGER.debug(f"{'Lane Poly and drawing:':<37}{Profiler.get_avg_millis('lane_drawing'):>6.3f} ms")
			LOGGER.debug(f"{'Detection Drawing:':<37}{Profiler.get_avg_millis('lane_drawing'):>6.3f} ms")
			LOGGER.debug(f"{'Cls Drawing:':<37}{Profiler.get_avg_millis('cls_drawing'):>6.3f} ms")

			break
		elif k == ord('c'):
			cv2.imwrite(f"result_{randint(0, 100)}.png", show_img)


	cap.release()
	writer.release()

if __name__ == '__main__':
	"""
	-c ../conf/experiments/efficientnetb2_fpn.json
	-w ../log/MT_ADASNET/efficientnetb2_fpn.2022.6.17.15.34.8.8uvc1bun/last.pth
	"""
	main()

