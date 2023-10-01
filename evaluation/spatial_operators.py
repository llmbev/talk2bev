import math
import numpy as np

# json_list = [{'object_id': 1, 'bev_centroid': [37, 132], 'matched_point': [[850.32800337], [509.03680517], [1]], 'matched_cam': 'CAM_BACK', 'llm_message': 'This is a view from the top of a hill looking down onto a stretch of highway with a single lane of traffic in each direction. The highway is lined with trees on either side and there are no buildings or other structures in the distance. The sky is clear and there are no clouds visible. The image is black and white.'}, {'object_id': 2, 'bev_centroid': [38, 136], 'matched_point': [[850.32800337], [509.03680517], [1]], 'matched_cam': 'CAM_BACK', 'llm_message': 'This image shows a street with no cars on it. The street is straight and has two white lines painted on it, one on the left side and one on the right side. There are no trees or other vegetation on the street, but there is some light poles on either side of the street. The sky is clear and there are no clouds in it. The horizon is visible in the distance.'}, {'object_id': 3, 'bev_centroid': [80, 115], 'matched_point': [[141011428000], [528.161582], [100000000]], 'matched_cam': 'CAM_BACK', 'llm_message': "The central object in this image is a black car with a sleek and elegant design. It has a low profile, with the rear end of the car sloping down towards the front. The car has large headlights on either side of the front bumper, and a set of fog lights on the lower part of the front bumper. The car's body is sleek and streamlined, with a set of curves that give it a smooth and aerodynamic look. The car has large alloy wheels with a shiny finish, and the tires are inflated to the appropriate pressure. The car's back end is also sleek, with a set of tail lights on either side of the rear bumper. The car is parked on the side of the road, with the driver's side door open and the keys in the ignition."}, {'object_id': 4, 'bev_centroid': [107, 137], 'matched_point': [[229.3340014], [525.64378004], [1]], 'matched_cam': 'CAM_FRONT_LEFT', 'llm_message': 'The image shows a silver car parked on a street. The car is parked facing the camera, with its hood and front bumper visible. There is a person standing behind the car, facing away from the camera, and holding onto the door handle. The person is wearing a black jacket, black pants, and a black backpack. There is a blue sky visible in the background, with some clouds visible.'}, {'object_id': 5, 'bev_centroid': [111, 91], 'matched_point': [[563.11477746], [6118195322], [1]], 'matched_cam': 'CAM_FRONT_RIGHT', 'llm_message': 'The central object in this image is a silver car with its door open, showing the interior of the car. The car is parked in a parking lot with other cars in the background.'}, {'object_id': 6, 'bev_centroid': [120, 142], 'matched_point': [[53103049032], [567.96627876], [1]], 'matched_cam': 'CAM_FRONT_LEFT', 'llm_message': 'The central object in this image is a small, red and white smart car. The car appears to be parked on the side of a road or street, with its rear end facing towards the viewer. The car has four wheels, a hood, a trunk, and a windshield. The body of the car is also white. The car appears to be new and well maintained. There are no people or other objects visible in the image.'}, {'object_id': 7, 'bev_centroid': [126, 109], 'matched_point': [[312.41813077], [610.587429], [1]], 'matched_cam': 'CAM_FRONT', 'llm_message': 'This image is a silver car with a sleek design. The car appears to be a sedan, with a hatchback and a large front grille. The windows are tinted and the headlights are bright. The wheels are also shiny and the car has a lot of curves and lines. The overall design of the car is very modern and sleek.'}, {'object_id': 8, 'bev_centroid': [137, 98], 'matched_point': [[919.16787331], [576.25356425], [1]], 'matched_cam': 'CAM_FRONT', 'llm_message': 'The central object in this image is a person walking on a sidewalk with a bag in hand. There are no other objects in the background.'}, {'object_id': 9, 'bev_centroid': [146, 75], 'matched_point': [[155568979000], [513.56061], [100000000]], 'matched_cam': 'CAM_FRONT', 'llm_message': 'The central object in this image is a pair of boots with a metal heel. The boots are black and appear to be made of some type of hard, durable material. The heel has a sharp point that is likely to be used for a variety of purposes, such as piercing or stabbing. The overall appearance of the boots is rugged and functional, with no unnecessary decoration or ornamentation. The image is dark, with only the boots visible against a black background. The lack of any other objects in the image makes it difficult to determine the context or purpose of the boots.'}, {'object_id': 10, 'bev_centroid': [168, 90], 'matched_point': [[103791306000], [525.333272], [100000000]], 'matched_cam': 'CAM_FRONT', 'llm_message': 'The central object in this image is a white car. It is parked on a black background. The car has a blue license plate on the rear bumper. There are no other visible details on the car.'}, {'object_id': 11, 'bev_centroid': [179, 75], 'matched_point': [[134221265000], [492.135352], [100000000]], 'matched_cam': 'CAM_FRONT', 'llm_message': 'The central object in this image is a silver car with a blue front bumper parked on the side of the road. It appears to have a dent on the front left side of the car. The car is facing towards the left side of the road, with its headlights illuminated. There are no other objects visible in the image.'}, {'object_id': 12, 'bev_centroid': [183, 112], 'matched_point': [[665.18007135], [472.56033511], [1]], 'matched_cam': 'CAM_FRONT', 'llm_message': "The central object in this image is a white refrigerated trailer with the words 'Johns Furniture' written on the side. The trailer is parked in front of a building with white and black stripes. There are no people or other vehicles in the image.\n\nThe trailer appears to be in good condition, with no visible dents or damage. It has a white body with a black roof and silver accents. The refrigeration unit on the back of the trailer is also white.\n\nThere is a small amount of debris on the ground in front of the trailer, including a few leaves and some pieces of paper. The image appears to have been taken at night, with the streetlights and some car headlights illuminating the scene. The weather appears to be overcast, with a few clouds visible in the sky.\n\nThe trailer is parked in front of a building with white and black stripes. It is not clear what is inside the building, but it appears to be a retail establishment of some kind. There are no other vehicles in the parking lot, and no people visible in the image.\n\nThe trailer appears to be a common type of refrigerated truck used for transporting goods such as furniture and appliances. It is equipped with a refrigeration unit that allows it to maintain a specific temperature inside"}, {'object_id': 13, 'bev_centroid': [186, 89], 'matched_point': [[112906119000], [496.954583], [100000000]], 'matched_cam': 'CAM_FRONT', 'llm_message': "The central object in this image is a car with its engine visible. The car is a black sedan with a white stripe on the side and a license plate on the front. There is a reflection in the windshield of a building in the background. The image appears to be taken at night with the headlights on. The car's engine appears to be visible and is not blurry. There is no other object in the image."}, {'object_id': 14, 'bev_centroid': [197, 116], 'matched_point': [[665.18007135], [472.56033511], [1]], 'matched_cam': 'CAM_FRONT', 'llm_message': "The central object in this image is a white box truck with a white roof, a black front bumper, and black tires. The truck has a white interior with several buttons and knobs on the dashboard. The words 'Garbage Truck' are written on the side of the truck in white letters. There is a small sign on the side of the truck that reads 'Please do not block the road.' The truck is parked on the side of the road with no other vehicles in the background. The sky is cloudy and gray. There are no people or other objects visible in the image."}, {'object_id': 15, 'bev_centroid': [199, 73], 'matched_point': [[123270456000], [495.200474], [100000000]], 'matched_cam': 'CAM_FRONT', 'llm_message': 'The image shows a close up of the head of a person wearing a hat with a dark bandana tied around the neck, with a large smile on their face. Their eyes are closed and they appear to be sleeping. The background is dark with a faint glow coming from a window in the corner of the room.'}, {'object_id': 16, 'bev_centroid': [0, 0], 'matched_point': [[812.12777078], [490.19599743], [1]], 'matched_cam': 'CAM_BACK', 'llm_message': 'The central object in this image is a roadway with no cars present. It appears to be a straight road with no turnoffs or intersections. The road is made of asphalt and there are no trees or other landscaping in the area. The sky is clear and there are no clouds visible. The horizon line is clearly visible in the distance, indicating that the road continues straight ahead.'}]

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
