# -*- coding: utf-8 -*-
# ---------------------

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import RANSACRegressor

import warnings
warnings.simplefilter('ignore', np.RankWarning)

def get_clusters(X, y):
	s = np.argsort(y)
	return np.split(X[s], np.unique(y[s], return_index=True)[1][1:])

class PolynomialRegression(object):
	def __init__(self, degree=2, coeffs=None):
		self.degree = degree
		self.coeffs = coeffs

	def fit(self, X, y):
		self.coeffs = np.polyfit(X.ravel(), y, self.degree)

	def get_params(self, deep=False):
		return {'coeffs': self.coeffs}

	def set_params(self, coeffs=None, random_state=None):
		self.coeffs = coeffs

	def predict(self, X):
		poly_eqn = np.poly1d(self.coeffs)
		y_hat = poly_eqn(X.ravel())
		return y_hat

	def score(self, X, y):
		return mean_squared_error(y, self.predict(X))

def cluster_lane_preds(lanes, lanes_cls, lanes_votes):
	lane_clusters = [[] for _ in range(8)]
	for lc in range(8):
		current_cls = lanes_cls.eq(lc).nonzero()
		lind = lanes[current_cls, :2].squeeze()
		votes = lanes_votes[:, current_cls].squeeze()

		if lind.shape[0] == 0 or len(lind.shape) != 2:
			continue

		votes = (votes.T + lind).cpu().numpy()
		clusters = AgglomerativeClustering(n_clusters=None,
		                                   distance_threshold=8.0 * 4, linkage='ward').fit_predict(votes)

		clusters = get_clusters(lind.cpu().numpy(), clusters)
		lane_clusters[lc] += clusters

	return lane_clusters

def fast_clustering(lanes, lanes_cls, lanes_votes):
	lane_clusters = [[] for _ in range(8)]
	for lc in range(8):
		current_cls = (lanes_cls == lc).nonzero()
		lind = lanes[current_cls, :2].squeeze()
		votes = lanes_votes[:, current_cls].squeeze()

		if lind.shape[0] == 0 or len(lind.shape) != 2:
			continue

		votes = (votes.T + lind)  # .cpu().numpy()
		clusters = AgglomerativeClustering(n_clusters=None,
		                                   distance_threshold=8.0 * 4, linkage='ward').fit_predict(votes)

		clusters = get_clusters(lind, clusters)
		lane_clusters[lc] += clusters
	return lane_clusters

def fit_lanes(lane_clusters):

	lanes_fitted = {i : [] for i in range(len(lane_clusters))}

	for cla, cls_clusters in enumerate(lane_clusters):
		for cl in cls_clusters:

			if cl.shape[0] < 5:
				continue

			x = cl[:, 0]
			y = cl[:, 1]

			ransac = RANSACRegressor(PolynomialRegression(degree=3),
			                         residual_threshold=0.5 * np.std(x),
			                         random_state=0)

			# calculate polynomial
			try:
				ransac.fit(np.expand_dims(x, axis=1), y)
			except ValueError:
				continue

			# calculate new x's and y's
			x_new = np.linspace(min(x), max(x), len(x))
			y_new = ransac.predict(np.expand_dims(x_new, axis=1))

			newlane = np.stack([x_new, y_new], axis=-1)
			lanes_fitted[cla].append(newlane)

	return lanes_fitted
