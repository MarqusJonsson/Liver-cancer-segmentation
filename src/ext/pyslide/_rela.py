import numpy as np
from shapely import geometry

__all__ = [	"cnt_inside_ratio",
			]

def cnt_inside_ratio(cnt_arr1, cnt_arr2):
	""" Calculate the ratio between intersection part of cnt_arr1 and cnt_arr2
	to cnt_arr1.

	Parameters
	-------
	cnt_arr1: np.array
		contour with standard numpy 2d array format
	cnt_arr2: np.array
		contour with standard numpy 2d array format

	Returns
	-------
	ratio: float
		intersection ratio of cnt_arr1

	"""

	# construct contour polygon
	point_list1, point_list2 = [], []
	num_point1 = cnt_arr1.shape[1]
	num_point2 = cnt_arr2.shape[1]
	# need to change h-w to w-h
	for ind in np.arange(num_point1):
		point_list1.append((cnt_arr1[1][ind], cnt_arr1[0][ind]))
	for ind in np.arange(num_point2):
		point_list2.append((cnt_arr2[1][ind], cnt_arr2[0][ind]))
	cnt_poly1 = geometry.Polygon(point_list1)
	cnt_poly1 = cnt_poly1.convex_hull
	cnt_poly2 = geometry.Polygon(point_list2)
	cnt_poly2 = cnt_poly2.convex_hull

	inter_flag = cnt_poly1.intersects(cnt_poly2)
	if inter_flag == False:
		ratio = 0.0
	else:
		inter_poly = cnt_poly1.intersection(cnt_poly2)
		inter_area = inter_poly.area
		cnt1_area = cnt_poly1.area
		ratio = inter_area * 1.0 / cnt1_area

	return ratio