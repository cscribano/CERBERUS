from pathlib import Path
import numpy as np
import numba as nb

from profiler import Profiler
from .trt_infer import TRTModel, TRTInference

def cerberus_model(model):

    class Cerberus(TRTModel):
        if model is None:
            ENGINE_PATH = Path(__file__).parent.parent.parent / 'weights' / 'last_sim.trt'
            MODEL_PATH = Path(__file__).parent.parent.parent / 'weights' / 'last_sim.onnx'
        else:
            MODEL_PATH = Path(model)
            ENGINE_PATH = Path(model).with_suffix('.trt')

        INPUT_SHAPE = (3, 320, 640)
        OUTPUT_LAYOUT = 1

    return Cerberus

class CerberusInference:
    def __init__(self, model=None):

        self.model = cerberus_model(model)
        self.batch_size = 1

        self.backend = TRTInference(self.model, 1)
        self.inp_handle = self.backend.input.host.reshape(*self.model.INPUT_SHAPE)

        self.preds = []

    def __call__(self, frame, raw=False):
        """Extract feature embeddings from bounding boxes synchronously."""
        self.extract_async(frame)
        return self.postprocess(raw)

    def extract_async(self, frame):
        # pipeline inference and preprocessing the next batch in parallel
        self._preprocess(frame)
        self.backend.infer_async()

    def postprocess(self, raw=False):
        """Synchronizes, applies postprocessing, and returns a NxM matrix of N
        extracted embeddings with dimension M.
        This API should be called after `extract_async`.
        """

        preds_out = self.backend.synchronize()

        if raw:
            return preds_out

        with Profiler('inference_decode'):
            ## Decode boxes
            d_offsets = preds_out[2].reshape(-1, 80, 160)
            d_heatmaps = preds_out[4].reshape(-1, 80, 160)
            d_occl = preds_out[3].reshape(-1, 80, 160)

            d_scores, d_indices, d_labels = self._decode_heatmap(d_heatmaps, th=0.6)
            d_occl = self._sigmoid(d_occl[0, d_indices[:, 1], d_indices[:, 0]])

            bb_ofs = d_offsets[:, d_indices[:, 1], d_indices[:, 0]]
            x1x2 = (bb_ofs[:2] + d_indices[..., 0][np.newaxis, :]) * 4
            y1y2 = (bb_ofs[2:] + d_indices[..., 1][np.newaxis, :]) * 4
            boxes = np.stack([x1x2[0], y1y2[0], x1x2[1], y1y2[1], d_scores, d_labels, d_occl], axis=-1)

            # Decode lanes
            l_heatmaps = preds_out[1].reshape(-1, 80, 160) # 8
            l_offsets = preds_out[0].reshape(-1, 80, 160) # 2

            l_scores, l_indices, l_labels = self._decode_heatmap(l_heatmaps, th=0.6)

            l_votes = l_offsets[:, l_indices[:, 1], l_indices[:, 0]] * 4
            l_indices = l_indices * 4
            lanes = np.concatenate([l_indices.astype(np.float32), l_scores[..., np.newaxis]], axis=-1)

            # Decode classification results
            cls = tuple(preds_out[5:])

        return boxes, (lanes, l_labels, l_votes), cls


    @staticmethod
    def _decode_heatmap(heatmap, th=0.6):
        labels = np.argmax(heatmap, axis=0)
        heatmap = np.take_along_axis(heatmap, labels[np.newaxis,], 0)[0]

        indices = np.stack(np.nonzero(heatmap > th), axis=-1)[:, ::-1]
        scores = heatmap[indices[:, 1], indices[:, 0]]
        labels = labels[indices[:, 1], indices[:, 0]]

        return scores, indices, labels

    def _preprocess(self, img):
        self._normalize(img, self.inp_handle)

    @staticmethod
    @nb.njit(fastmath=True, nogil=True, cache=True)
    def _normalize(img, out):
        # HWC -> CHW
        chw = img.transpose(2, 0, 1)
        # Normalize using ImageNet's mean and std
        out[0, ...] = (chw[0, ...] / 255. - 0.485) / 0.229
        out[1, ...] = (chw[1, ...] / 255. - 0.456) / 0.224
        out[2, ...] = (chw[2, ...] / 255. - 0.406) / 0.225
    @staticmethod
    @nb.njit(fastmath=True, nogil=True, cache=True)
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

if __name__ == '__main__':
    cb = CerberusInference()

    mdt = 0
    for _ in range(100):
        src = np.random.rand(320, 640, 3)
        y = cb(src)
        dt = cb.backend.get_infer_time()
        mdt += dt
    print(mdt/100)
