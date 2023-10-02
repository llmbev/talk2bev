import numpy as np
import re
import json
import decimal
import os

from spatial_operators import *

s1 = 0
s2 = 0
dir1 = 'spatial_gt/' # path to GT jsons
dir2 = 'spatial_pred/' # path to answer jsons

files = os.listdir(dir1)

for fll in files:
    key = fll[:-5]

    json_gt = json.load(open(dir1[8:]+key+'.json'))
    json_pd = json.load(open(dir2[8:]+key+'.json'))

    # Question 1
    d1 = get_object_dist(json_gt, json_gt[0]['token'][0])
    d2 = get_object_dist(json_pd, json_gt[0]['token'][0])
    s1 += dist_score(d1, d2)

    # Question 2
    d1 = get_object_dist(json_gt, json_gt[1]['token'][0])
    d2 = get_object_dist(json_pd, json_gt[1]['token'][0])
    s1 += dist_score(d1, d2)

    # Question 3
    d1 = distance_between_objects(json_gt, json_gt[0]['token'][0], json_gt[1]['token'][0])
    d2 = distance_between_objects(json_pd, json_gt[0]['token'][0], json_gt[1]['token'][0])
    s1 += dist_score(d1, d2)

    # Question 4
    jgt = get_k_closest_jsons(json_gt, 2)
    jpd = get_k_closest_jsons(json_pd, 2)
    d1 = distance_between_objects(json_gt, jgt[0]['token'][0], jgt[1]['token'][0])
    d2 = distance_between_objects(json_pd, jpd[0]['token'][0], jpd[1]['token'][0])
    s1 += dist_score(d1, d2)

    dd = 20

    # Question 5
    jgt = distance_filtering(json_gt, dd)
    jpd = distance_filtering(json_pd, dd)
    s2 += iou(jgt, jpd)

    # Question 6
    jgt = get_k_closest_jsons(json_gt, 3)
    jpd = get_k_closest_jsons(json_pd, 3)
    s2 += iou(jgt, jpd)

    # Question 7
    jgt = filter_front(json_gt)
    jgt = distance_filtering(jgt, dd)
    jpd = filter_front(json_pd)
    jpd = distance_filtering(jpd, dd)
    s2 += iou(jgt, jpd)

    # Question 8
    jgt = filter_rear(json_gt)
    jgt = get_k_closest_jsons(jgt, 3)
    jpd = filter_rear(json_pd)
    jpd = get_k_closest_jsons(jpd, 3)
    s2 += iou(jgt, jpd)

s1 = s1/4/7
s1 = decimal.Decimal(s1)
s1 = s1.quantize(decimal.Decimal('0.00'))

s2 = s2/4/7
s2 = decimal.Decimal(s2)
s2 = s2.quantize(decimal.Decimal('0.00'))

print("IoU:", s2)
print("Distance Score:", s1)
