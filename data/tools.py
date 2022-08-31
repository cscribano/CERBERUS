# -*- coding: utf-8 -*-
# ---------------------

import numpy as np

from scipy.special import comb
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def bernstein_poly(i, n, t):
	"""
	 The Bernstein polynomial of n, i as a function of t
	"""

	return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
	"""
	   Given a set of control points, return the
	   bezier curve defined by the control points.

	   points should be a list of lists, or list of tuples
	   such as [ [1,1],
				 [2,3],
				 [4,5], ..[Xn, Yn] ]
		nTimes is the number of time steps, defaults to 1000

		See http://processingjs.nihongoresources.com/bezierinfo/
	"""

	nPoints = len(points)
	xPoints = np.array([p[0] for p in points])
	yPoints = np.array([p[1] for p in points])

	t = np.linspace(0.0, 1.0, nTimes)

	polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

	xvals = np.dot(xPoints, polynomial_array)
	yvals = np.dot(yPoints, polynomial_array)

	return xvals, yvals

def get_bezier_parameters(X, Y, degree=3):
	""" Least square qbezier fit using penrose pseudoinverse.

	Parameters:

	X: array of x data.
	Y: array of y data. Y[0] is the y point for X[0].
	degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.

	Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
	and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
	"""
	if degree < 1:
		raise ValueError('degree must be 1 or greater.')

	if len(X) != len(Y):
		raise ValueError('X and Y must be of the same length.')

	if len(X) < degree + 1:
		raise ValueError(f'There must be at least {degree + 1} points to '
		                 f'determine the parameters of a degree {degree} curve. '
		                 f'Got only {len(X)} points.')

	def bpoly(n, t, k):
		""" Bernstein polynomial when a = 0 and b = 1. """
		return t ** k * (1 - t) ** (n - k) * comb(n, k)

	# return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

	def bmatrix(T):
		""" Bernstein matrix for Bézier curves. """
		return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])

	def least_square_fit(points, M):
		M_ = np.linalg.pinv(M)
		return M_ * points

	T = np.linspace(0, 1, len(X))
	M = bmatrix(T)
	points = np.array(list(zip(X, Y)))

	final = least_square_fit(points, M).tolist()
	final[0] = [X[0], Y[0]]
	final[len(final) - 1] = [X[len(X) - 1], Y[len(Y) - 1]]
	return final


def compare_labels(l1, l2):

	pt1 = l1["keypoints"]
	pt2 = l2["keypoints"]

	l = max(len(pt1), len(pt2))
	assert l >= 3

	if len(pt1) == l:
		pts = l2["poly2d"][0]['vertices']
		xvals, yvals = bezier_curve(pts, nTimes=l)
		pt2 = np.stack([xvals, yvals], axis=-1)
	else:
		pts = l1["poly2d"][0]['vertices']
		xvals, yvals = bezier_curve(pts, nTimes=l)
		pt1 = np.stack([xvals, yvals], axis=-1)

	pt1, pt2 = np.array(pt1), np.array(pt2)
	closest = cdist(pt1, pt2).argmin(0)

	return pt1, pt2[closest]

def dist(k1, k2):

	if k1 is None or k2 is None:
		return 1e5

	if k1['id'] == k2['id']:
		return 1e5

	c1 = (k1['attributes']['laneDirection'] == k1['attributes']['laneDirection'] == 'parallel')
	c2 = k1['category'] == k2['category']
	c3 =True #"double" not in k1['category']

	if not (c1 and c2 and c3):
		return 1e5

	pt1, pt2 = compare_labels(k1, k2)
	dist = np.linalg.norm(pt1 - pt2, axis=-1).mean()

	return dist
