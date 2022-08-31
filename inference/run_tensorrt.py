# -*- coding: utf-8 -*-
# ---------------------

import click

import cv2
import numpy as np
import logging
from time import time, sleep

from trt_inference.cerberus_trt import CerberusInference
from trt_inference.cls import WTR_CLS, SN_CLS, TD_CLS, DET_CLS_IND
from postproc import get_clusters, fast_clustering

from profiler import Profiler

logging.basicConfig(format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
Profiler.set_warmup(25)

cls_col = [(153, 255, 102), (255, 255, 255), (0, 255, 255), (52, 255, 52), (51, 153, 51),
           (0, 255, 0), (153, 51, 51), (0, 0, 255), (255, 0, 0)]

lancol = [(0, 255, 255), (255, 255, 255), (255, 150, 50), (0, 0, 255),
          (102, 0, 102), (10, 255, 0), (255, 255, 0), (0, 153, 255)]

@click.command()
@click.option('--model_file', '-m', type=click.Path(exists=True), default=None, required=False)
@click.option('--video', '-v', type=click.Path(exists=True), default=None, required=True)
@click.option('--max_frames', '-f', type=int, default=None, required=False)
@click.option('--infer_only', '-i', type=click.BOOL, default=False, required=False)
def main(model_file, video, max_frames, infer_only):

	# load video
	# video = "../videos/dashcam_demo.mp4"
	cap = cv2.VideoCapture(video)
	cap.set(cv2.CAP_PROP_POS_FRAMES, 150 * 30)

	"""# writer
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	writer = cv2.VideoWriter('../videos/result_trt.avi', fourcc, 80, (640, 320))"""

	# Classes
	wtr = {v: k for k, v in WTR_CLS.items()}
	scn = {v: k for k, v in SN_CLS.items()}
	td = {v: k for k, v in TD_CLS.items()}

	model = CerberusInference(model_file)

	times = []
	infer_times = []
	frames = 0
	while cap.isOpened():
		t = time()

		with Profiler('acquire'):
			_, frame = cap.read()
			frame = cv2.resize(frame, (640, 360))
			frame = frame[20:340, :, :]

		with Profiler('inference_all'):
			preds = model(frame, raw=infer_only)

		it = model.backend.get_infer_time()
		infer_times.append(it)

		if not infer_only:

			det_out, lane_out, scn_out = preds
			boxes = det_out
			lanes, lanes_cls, lanes_votes = lane_out

			# Classification results
			w_cls = wtr[scn_out[0].item()]
			s_cls = scn[scn_out[1].item()]
			td_cls = td[scn_out[2].item()]

			# Lane clustering
			with Profiler('lane_clustering'):
				lane_clusters = fast_clustering(lanes, lanes_cls, lanes_votes)

			# Draw keypoints
			with Profiler('lane_drawing'):
				for cla, cls_clusters in enumerate(lane_clusters):
					for cl in cls_clusters:

						col = lancol[cla]
						if cl.shape[0] < 5:
							continue

						x = cl[:, 0]
						y = cl[:, 1]

						# calculate polynomial
						try:
							z = np.polyfit(x, y, 2)
							f = np.poly1d(z)
						except ValueError:
							continue

						# calculate new x's and y's
						x_new = np.linspace(min(x), max(x), len(x) * 2)
						y_new = f(x_new)

						for cx, cy in zip(x_new, y_new):
							frame = cv2.circle(frame, (int(cx), int(cy)), 1, col, thickness=2, )

			# Draw boxes
			with Profiler('det_drawing'):
				for b in boxes:
					cls = DET_CLS_IND[int(b[5])].split(" ")[-1]
					tl = (int(b[2]), int(b[3]))
					br = (int(b[0]), int(b[1]))

					color = (0, 255, 0) if b[6] < 0.5 else (0,0,255)
					cv2.rectangle(frame, tl, br, color, 2)

					(text_width, text_height), _ = cv2.getTextSize(cls, cv2.FONT_HERSHEY_DUPLEX, 0.3, 1)
					cv2.rectangle(frame, br, (br[0] + text_width - 1, br[1] + text_height - 1),
					              color, cv2.FILLED)
					cv2.putText(frame, cls, (br[0], br[1] + text_height - 1), cv2.FONT_HERSHEY_DUPLEX,
					            0.3, 0, 1, cv2.LINE_AA)

			# Add text
			with Profiler('cls_drawing'):
				text = f"WEATHER: {w_cls}   SCENE: {s_cls}   DAYTIME: {td_cls}"
				frame = cv2.rectangle(frame, (10, 5), (550, 25), (0, 0, 0), -1)
				frame = cv2.putText(frame, text, (15, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5,
				                    (255,255,255), 1, cv2.LINE_AA, False)

			# writer.write(frame)
			cv2.imshow("result", frame)

		dt = time() - t
		times.append(dt)
		frames +=1

		if cv2.waitKey(1) == ord('q') or (frames == max_frames):
			print('=================Timing Stats=================')
			print(f"{'Frame Acquiring:':<37}{Profiler.get_avg_millis('acquire'):>6.3f} ms")
			print(f"{'Inference total:':<37}{Profiler.get_avg_millis('inference_all'):>6.3f} ms")
			print(f"\t{'Inference DNN:':<37}{np.array(infer_times[10:]).mean():>6.3f} ms")
			print(f"\t{'Inference Decoding:':<37}{Profiler.get_avg_millis('inference_decode'):>6.3f} ms")
			print('----------------------------------------------')
			print(f"{'Lanes clustering:':<37}{Profiler.get_avg_millis('lane_clustering'):>6.3f} ms")
			print(f"{'Lanes Fitting and Drawing:':<37}{Profiler.get_avg_millis('lane_drawing'):>6.3f} ms")
			print(f"{'Detection Drawing:':<37}{Profiler.get_avg_millis('det_drawing'):>6.3f} ms")
			print(f"{'Cls Drawing:':<37}{Profiler.get_avg_millis('cls_drawing'):>6.3f} ms")
			print(f"{'AVERAGE TIME:':<37}{np.array(times[10:]).mean()*1000:>6.3f} ms")
			break

	cap.release()
	# writer.release()


if __name__ == '__main__':
	main()
