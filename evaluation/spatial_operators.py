import math
import heapq
import numpy as np

main_coverage_angle = 90
diagonal_coverage_angle = 90

def calculate_angle(obj_x, obj_y):
    dx1, dy1 = 100, 0
    dx2, dy2 = obj_x - 100, obj_y - 100
    dot_product = dx1 * dx2 + dy1 * dy2
    magnitude_v1 = math.sqrt(dx1**2 + dy1**2)
    magnitude_v2 = math.sqrt(dx2**2 + dy2**2)
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle_radians = math.acos(cosine_angle)
    angle_degrees = math.degrees(angle_radians)
    cross_product = dx1 * dy2 - dx2 * dy1
    if cross_product < 0:
        angle_degrees = -angle_degrees
    return angle_degrees
def filter_rear(json_list):
    rear_json = []
    for gg in json_list:
        obj_x, obj_y = gg['bev_centroid']
        angle = calculate_angle(obj_x, obj_y)
        if angle >= (180 - main_coverage_angle/2) or angle <= (-180 + main_coverage_angle/2):
            rear_json.append(gg)
    return rear_json
def filter_left(json_list):
    left_json = []
    for gg in json_list:
        obj_x, obj_y = gg['bev_centroid']
        angle = calculate_angle(obj_x, obj_y)
        if angle >= (90 - main_coverage_angle/2) and angle <= (90 + main_coverage_angle/2):
            left_json.append(gg)
    return left_json
def filter_front(json_list):
    front_json = []
    for gg in json_list:
        obj_x, obj_y = gg['bev_centroid']
        angle = calculate_angle(obj_x, obj_y)
        if angle >= -main_coverage_angle/2 and angle <= main_coverage_angle/2:
            front_json.append(gg)
    return front_json
def filter_right(json_list):
    right_json = []
    for gg in json_list:
        obj_x, obj_y = gg['bev_centroid']
        angle = calculate_angle(obj_x, obj_y)
        if angle >= (-90 - main_coverage_angle/2) and angle <= (-90 + main_coverage_angle/2):
            right_json.append(gg)
    return right_json
def distance_filtering(json_list, dist):
    dist_px = 2*dist
    filtered_json = []
    for gg in json_list:
        obj_x, obj_y = gg['bev_centroid']
        if math.sqrt((obj_x-100)**2 + (obj_y-100)**2) <= dist_px:
            filtered_json.append(gg)
    return filtered_json
def distance_between_objects(json_list, id1, id2):
    x1, y1 = -1, -1
    x2, y2 = -1, -1
    for gg in json_list:
        if gg['token'][0]==id1:
            x1, y1 = gg['bev_centroid']
        if gg['token'][0]==id2:
            x2, y2 = gg['bev_centroid']
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)
def get_object_dist(json_list, object_id):
    x, y = -1, -1
    for gg in json_list:
        if gg['token'][0]==object_id:
            x, y = gg['bev_centroid']
            # print(object_id, x, y)
    return math.sqrt((x-100)**2 + (y-100)**2)
def get_k_closest_jsons(json_list, k):    
    filtered_json = []
    # print("---")
    for gg in json_list:
        x, y = gg['bev_centroid']
        distance = math.sqrt((x-100)**2 + (y-100)**2)
        # print(distance, gg['token'][0])
        heapq.heappush(filtered_json, (-distance, gg))
        if len(filtered_json) > k:
            heapq.heappop(filtered_json)
    return [json_data for (_, json_data) in filtered_json]
def get_k_farthest_jsons(json_list, k):    
    filtered_json = []
    for gg in json_list:
        x, y = gg['bev_centroid']
        distance = math.sqrt((x-100)**2 + (y-100)**2)
        heapq.heappush(filtered_json, (distance, gg))
        if len(filtered_json) > k:
            heapq.heappop(filtered_json)
    return [json_data for (_, json_data) in filtered_json]


##############################
######### Evaluation #########
##############################


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

def evaluate_spatial_query(query_result, ground_truth):
    
    query_result_ids = [json_str["object_id"] for json_str in query_result]
    ground_truth_ids = [json_str["object_id"] for json_str in ground_truth]

    if len(query_result_ids) == 0 and len(ground_truth_ids) == 0:
        return 1
    
    else:
        intersection = len(set(query_result_ids) & set(ground_truth_ids))
        union = len(set(query_result_ids) | set(ground_truth_ids))
        iou = intersection / union
        return iou

# Distance should be in meters
def calculate_score(d1, d2):
    max_difference = 100 * np.sqrt(2)
    difference = abs(d1 - d2)
    if difference >= max_difference:
        score = 0
    else:
        score = 1 - difference / max_difference
    return score

main_coverage_angle = 90

def calculate_angle(obj_x, obj_y, org_x, org_y):
    dx1, dy1 = 200 - org_x, 0
    dx2, dy2 = obj_x - org_x, obj_y - org_y
    dot_product = dx1 * dx2 + dy1 * dy2
    magnitude_v1 = math.sqrt(dx1**2 + dy1**2)
    magnitude_v2 = math.sqrt(dx2**2 + dy2**2)
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle_radians = math.acos(cosine_angle)
    angle_degrees = math.degrees(angle_radians)
    cross_product = dx1 * dy2 - dx2 * dy1

    if cross_product < 0:
        angle_degrees = -angle_degrees

    return angle_degrees

def filter_rear(json_list, object_id):
    rear_json = []
    org_x, org_y = 100, 100
    
    if object_id != 0:      #passing object_id as 0 for ego vehicle
        for obj in json_list:
            if obj["object_id"] == object_id:
                org_x, org_y = obj['bev_centroid']

    for obj in json_list: 
        if obj['object_id'] != object_id:
            obj_x, obj_y = obj['bev_centroid']
            angle = calculate_angle(obj_x, obj_y, org_x, org_y)
            if angle >= (180 - main_coverage_angle/2) or angle <= (-180 + main_coverage_angle/2):
                rear_json.append(obj)
    return rear_json

def filter_left(json_list, object_id):
    left_json = []
    org_x, org_y = 100, 100
    
    if object_id != 0:      #passing object_id as 0 for ego vehicle
        for obj in json_list:
            if obj["object_id"] == object_id:
                org_x, org_y = obj['bev_centroid']

    for obj in json_list:
        if obj['object_id'] != object_id:
            obj_x, obj_y = obj['bev_centroid']
            angle = calculate_angle(obj_x, obj_y, org_x, org_y)
            if angle >= (90 - main_coverage_angle/2) and angle <= (90 + main_coverage_angle/2):
                left_json.append(obj)
    return left_json

def filter_front(json_list, object_id):
    front_json = []
    org_x, org_y = 100, 100
    
    if object_id != 0:      #passing object_id as 0 for ego vehicle
        for obj in json_list:
            if obj["object_id"] == object_id:
                org_x, org_y = obj['bev_centroid']

    for obj in json_list:
        if obj['object_id'] != object_id:
            obj_x, obj_y = obj['bev_centroid']
            angle = calculate_angle(obj_x, obj_y, org_x, org_y)
            if angle >= -main_coverage_angle/2 and angle <= main_coverage_angle/2:
                front_json.append(obj)
    return front_json

def filter_right(json_list, object_id):
    right_json = []
    org_x, org_y = 100, 100
    
    if object_id != 0:      #passing object_id as 0 for ego vehicle
        for obj in json_list:
            if obj["object_id"] == object_id:
                org_x, org_y = obj['bev_centroid']

    for obj in json_list:
        if obj['object_id'] != object_id:
            obj_x, obj_y = obj['bev_centroid']
            angle = calculate_angle(obj_x, obj_y, org_x, org_y)
            if angle >= (-90 - main_coverage_angle/2) and angle <= (-90 + main_coverage_angle/2):
                right_json.append(obj)
    return right_json

#assuming distance in meters
def find_distance(json_list, object_id1, object_id2):
    if object_id1 == 0:
        x1, y1 = 100, 100
    if object_id2 == 0:
        x2, y2 = 100, 100

    for obj in json_list:
        if obj['object_id'] == object_id1:
            x1, y1 = obj['bev_centroid']
        if obj['object_id'] == object_id2:
            x2, y2 = obj['bev_centroid']
    
    dist = (math.sqrt((x1-x2)**2 + (y1-y2)**2)) / 2
    return dist

def find_objects_within_distance(json_list, object_id, distance):
    nearby_json = []

    for obj in json_list:
        if obj['object_id'] != object_id:
            dist = find_distance(json_list, object_id, obj['object_id'])
            if dist <= distance:
                nearby_json.append(obj)
    return nearby_json

def get_k_closest_objects(json_list, object_id, k):
    closest_json = []
    closest_dist = []
    objects = []

    for obj in json_list:
        if obj['object_id'] != object_id:
            dist = find_distance(json_list, object_id, obj['object_id'])
            objects.append((obj, dist))
    closest_objects = sorted(objects, key=lambda obj: obj[1])

    for i in range(k):
        closest_json.append(closest_objects[i][0])
        closest_dist.append(closest_objects[i][1])

    return closest_json, closest_dist

def get_k_farthest_objects(json_list, object_id, k):
    farthest_json = []
    farthest_dist = []
    objects = []

    for obj in json_list:
        if obj['object_id'] != object_id:
            dist = find_distance(json_list, object_id, obj['object_id'])
            objects.append((obj, dist))
    farthest_objects = sorted(objects, key=lambda obj: obj[1], reverse=True)

    for i in range(k):
        farthest_json.append(farthest_objects[i][0])
        farthest_dist.append(farthest_objects[i][1])

    return farthest_json, farthest_dist

def filter_objects_with_tag(json_list, object_id, tagname, distance=1000):
    type_json = []

    nearby_json = find_objects_within_distance(json_list, object_id, distance)
    for obj in nearby_json:
        if obj['tag'] == tagname:
            type_json.append(obj)
    return type_json

def filter_color(json_list, object_id, color, distance=1000):
    type_json = []

    nearby_json = find_objects_within_distance(json_list, object_id, distance)
    for obj in nearby_json:
        if obj['color'] == color:
            type_json.append(obj)
    return type_json

def filter_size(json_list, object_id, distance, min_size, max_size):
    size_json = []

    nearby_json = find_objects_within_distance(json_list, object_id, distance)
    for obj in nearby_json:
        if obj['size'] >= min_size and obj['size'] <= max_size:
            size_json.append(obj)
    return size_json

#finds direction of object_id2 with respect to object_id1
def find_direction_of_object(json_list, object_id1, object_id2):
    org_x, org_y = 100, 100

    if object_id1 != 0:      #passing object_id as 0 for ego vehicle
        for obj in json_list:
            if obj["object_id"] == object_id1:
                org_x, org_y = obj['bev_centroid']

    for obj in json_list: 
        if obj['object_id'] == object_id2:
            obj_x, obj_y = obj['bev_centroid']
            angle = calculate_angle(obj_x, obj_y, org_x, org_y)
            if angle >= (180 - main_coverage_angle/2) or angle <= (-180 + main_coverage_angle/2):
                return 'REAR'
            elif angle >= (90 - main_coverage_angle/2) and angle <= (90 + main_coverage_angle/2):
                return 'LEFT'
            elif angle >= -main_coverage_angle/2 and angle <= main_coverage_angle/2:
                return 'FRONT'
            else:
                return 'RIGHT'



# frontFilter(objs)
# leftFilter(objs)
# rightFilter(objs)
# rearFilter(objs)
# distFilter(objs, X)
# objDistance(objs, id)
# kClosest(objs, k)
# kFarthest(objs, k)
# findDist(objs, id1, id2)
# objsInDist(objs, id, dist)
# kClosestToObj(objs, id, k)
# kFarthestToObj(objs, id, k)