import numpy as np

def dist_score(d1, d2):
    max_difference = 200 * np.sqrt(2)
    difference = abs(d1 - d2)
    if difference >= max_difference:
        score = 0
    else:
        score = 1 - difference / max_difference
    return score

def iou(query_result, ground_truth):
    query_result_ids = [json_str["token"][0] for json_str in query_result]
    ground_truth_ids = [json_str["object_id"] for json_str in ground_truth]
    if len(query_result_ids) == 0 and len(ground_truth_ids) == 0:
        return 1
    else:
        intersection = len(set(query_result_ids) & set(ground_truth_ids))
        union = len(set(query_result_ids) | set(ground_truth_ids))
        iou = intersection / union
        return iou