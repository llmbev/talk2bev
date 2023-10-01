import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from transformers import TextStreamer
# Load model directly
# Use a pipeline as a high-level helper
from transformers import pipeline

from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from nuscenes.utils.data_classes import PointCloud, LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--sam-checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="path to sam weights.")
    parser.add_argument("--model-path", type=str, default="/raid/t1/scratch/vikrant.dewangan/LLaVA/ckpt-old/", help="path to llava weights.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--conv-mode", type=int, default="llava_v1", help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def distance(x1, y1, z1, x2, y2, z2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + 0 * (z1 - z2) ** 2) ** ( 1 / 2)
 
# Function to calculate K closest points
def kClosest(points, target, K):
    pts = []
    n = len(points)
    d = []

    for i in range(n):
        d.append({
            "first": distance(points[i][0], points[i][1], points[i][2], target[0], target[1], target[2]),
            "second": i
        })
     
    d = sorted(d, key=lambda l:l["first"])
 
    for i in range(K):
        pt = []
        pt.append(points[d[i]["second"]][0])
        pt.append(points[d[i]["second"]][1])
        pt.append(points[d[i]["second"]][2])
        pt.append(points[d[i]["second"]][3])
        pts.append(pt)
 
    return pts

def get_image_projected_points(cam_left, cam_front, cam_right, cam_rear_left, cam_rear, cam_rear_right, arr):
    """
        arr: points to be back-projected: (K closest points)
    """
    cam_imgs = [cam_left, cam_front, cam_right, cam_rear_left, cam_rear, cam_rear_right]
    cam_keys = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    calibration_data = {
        'CAM_FRONT_LEFT':{
            'translation': [-1.57525595, -0.50051938, -1.50696033],
            'rotation': [[ 0.82254604, -0.56868433, -0.00401771], [ 0.00647832,  0.01643407, -0.99984396], [ 0.56866162,  0.82239167,  0.01720189]],
            'camera_intrinsic': [[1257.8625342125129, 0.0, 827.2410631095686], [0.0, 1257.8625342125129, 450.915498205774], [0.0, 0.0, 1.0]]
        },
        'CAM_FRONT':{
            'translation': [-1.72200568, -0.00475453, -1.49491292],
            'rotation': [[ 0.01026021, -0.99987258, -0.01222952], [ 0.00843345,  0.01231626, -0.99988859], [ 0.9999118 ,  0.01015593,  0.00855874]],
            'camera_intrinsic': [[1252.8131021185304, 0.0, 826.588114781398], [0.0, 1252.8131021185304, 469.9846626224581], [0.0, 0.0, 1.0]]
        },
        'CAM_FRONT_RIGHT':{
            'translation': [-1.58082566,  0.49907871, -1.51749368],
            'rotation': [[-0.84397973, -0.53614138, -0.01583178], [ 0.01645551,  0.00362107, -0.99985804], [ 0.5361226 , -0.84412044,  0.00576637]],
            'camera_intrinsic': [[1256.7485116440405, 0.0, 817.7887570959712], [0.0, 1256.7485116440403, 451.9541780095127], [0.0, 0.0, 1.0]]
        },
        'CAM_BACK_LEFT':{
            'translation': [-1.035691  , -0.48479503, -1.59097015],
            'rotation': [[ 0.94776036,  0.31896113,  0.00375564], [ 0.00866572, -0.0139763 , -0.99986478], [-0.31886551,  0.94766474, -0.01601021]],
            'camera_intrinsic': [[1256.7414812095406, 0.0, 792.1125740759628], [0.0, 1256.7414812095406, 492.7757465151356], [0.0, 0.0, 1.0]]
        },
        'CAM_BACK':{
            'translation': [-0.02832603, -0.00345137, -1.57910346],
            'rotation': [[ 0.00242171,  0.99998907, -0.00400023], [-0.01675361, -0.00395911, -0.99985181], [-0.99985672,  0.00248837,  0.01674384]],
            'camera_intrinsic': [[809.2209905677063, 0.0, 829.2196003259838], [0.0, 809.2209905677063, 481.77842384512485], [0.0, 0.0, 1.0]]
        },
        'CAM_BACK_RIGHT':{
            'translation': [-1.0148781 ,  0.48056822, -1.56239545],
            'rotation': [[-0.93477554,  0.35507456, -0.01080503], [ 0.01587584,  0.0113705 , -0.99980932], [-0.35488399, -0.93476883, -0.01626597]],
            'camera_intrinsic': [[1259.5137405846733, 0.0, 807.2529053838625], [0.0, 1259.5137405846733, 501.19579884916527], [0.0, 0.0, 1.0]]
        }
    }

    min_dist = 1.0
    flag = 0
    cam_img = None
    for cam_ind, cam_key in enumerate(cam_keys):
        barr = np.copy(arr)
        ppc = LidarPointCloud(barr.T);
        ppc.translate(np.array(calibration_data[cam_key]['translation']))
        ppc.rotate(np.array(calibration_data[cam_key]['rotation']))
        ddepths_img = ppc.points[2, :]
        points = view_points(ppc.points[:3, :], np.array(calibration_data[cam_key]['camera_intrinsic']), normalize=True)
        print(points)
        mask = np.ones(ddepths_img.shape[0], dtype=bool)
        mask = np.logical_and(mask, ddepths_img > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < 1600 - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < 900 - 1)

        print(mask)
        if mask.sum() > 0:
            # found
            flag = 1
            points = points[:, mask]
            ddepths_img = ddepths_img[mask]
            cam_img = cam_imgs[cam_ind]
            print(points.shape, " points")
            break

        if flag == 0:
            # no point able to back-project, just use front cam
            print("AAAAAAAAA no point found")
            cam_img = cam_front
    return cam_img

# ========================================
#             Model Initialization
# ========================================


print('Initializing Chat')
args = parse_args()
cfg = Config(args)

args.image_file = "/home/t1/vikrant.dewangan/llm-bev/MiniGPT-4/sample.jpg"
args.model_base = None

################ LLAVA #############
# model_name = get_model_name_from_path(args.model_path)
model_name = "llava"
tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, True, True)

if 'llama-2' in model_name.lower():
    conv_mode = "llava_llama_2"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"
print('Initialization Finished')

if args.conv_mode is not None and conv_mode != args.conv_mode:
    print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
else:
    args.conv_mode = conv_mode
conv = conv_templates[args.conv_mode].copy()
if "mpt" in model_name.lower():
    roles = ('user', 'assistant')
else:
    roles = conv.roles

####################################


# ========================================
#             Sam Initialization
# ========================================

from segment_anything import sam_model_registry, SamPredictor

model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(img_list):
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True)

def upload_img(gr_img, text_input):
    if gr_img is None:
        return None, None, gr.update(interactive=True), None
    image = gr_img
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(device)
    image = None
    return image_tensor, gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False)

def gradio_ask(user_message, image, chatbot):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + user_message
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + user_message
    conv.append_message(conv.roles[0], inp)
    chatbot = chatbot + [[user_message, None]]
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    return image, streamer, stopping_criteria, input_ids, chatbot

def gradio_answer(input_ids, streamer, stopping_criteria, image_tensor, chatbot):
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=1.0,
        max_new_tokens=1024,
        streamer=streamer,
        use_cache=True,
        stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    chatbot[-1][1] = outputs
    return chatbot

title = """<h1 align="center">Talk2BEV: Demo</h1>"""
description = """<h3>Upload BEV, perspective images, LiDAR point cloud and start chatting!</h3>"""
article = """<p><a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p><a href='https://raw.githubusercontent.com/Vision-CAIR/MiniGPT-4/main/MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>
"""

#TODO show examples below

def find_bounding_box(image):
    # Find the coordinates of non-zero (foreground) pixels along each channel
    foreground_pixels = np.array(np.where(np.any(image != 0, axis=2)))

    # Calculate the bounding box coordinates
    min_y, min_x = np.min(foreground_pixels, axis=1)
    max_y, max_x = np.max(foreground_pixels, axis=1)

    return min_y, min_x, max_y, max_x

def crop_around_bounding_box(image):
    min_y, min_x, max_y, max_x = find_bounding_box(image)

    # Crop the image using the bounding box coordinates
    cropped_image = image[min_y:max_y+1, min_x:max_x+1, :]

    return cropped_image

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    # gr.Markdown(article)

    image_tensor = gr.State()
    streamer = gr.State()
    stopping_criteria = gr.State()
    img = gr.State()
    input_ids = gr.State()

    tolerance = gr.Slider(label="Tolerance", info="How different colors can be in a segment.", minimum=0, maximum=256*3, value=50)

    with gr.Row():
        cam_left = gr.Image(label="CAM_LEFT - Upload")
        cam_front = gr.Image(label="CAM_FRONT - Upload")
        cam_right = gr.Image(label="CAM_RIGHT - Upload")
    
    with gr.Row():
        cam_rear_left = gr.Image(label="CAM_REAR_LEFT - Upload")
        cam_rear = gr.Image(label="CAM_REAR - Upload")
        cam_rear_right = gr.Image(label="CAM_REAR_RIGHT - Upload")

    with gr.Row():
        bev_img = gr.Image(label="BEV Image - Click a segment")
        image = gr.Image(label="Selected Segment", type="pil")                    
        lidar_data = gr.File(label="LiDAR Data - Upload a file")

    with gr.Row():
        upload_button = gr.Button(value="Submit Mask & Start Chat", interactive=True, variant="primary")
        clear = gr.Button("Clear All")

    with gr.Row():
        num_beams = gr.Slider(
            minimum=1,
            maximum=10,
            value=1,
            step=1,
            interactive=True,
            label="beam search numbers)",
        )
        
        temperature = gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Temperature",
        )

    with gr.Row():
        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='LLava')
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)

    def get_select_coords(cam_left, cam_front, cam_right, cam_rear_left, cam_rear, cam_rear_right, bev, lidar, tolerance, evt: gr.SelectData):
        lidardata = np.load(lidar.name)
        # import pdb; pdb.set_trace()
        print(evt.index[1], evt.index[0])
        print(bev[evt.index[1], evt.index[0]])

        labels_allowed_seg = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        labels_allowed_hdmap = [24, 25, 26, 27]
        print(np.array(bev[evt.index[1], evt.index[0]]).shape, np.array([0,0,255]).shape)
        if bev[evt.index[1], evt.index[0]][0] == 0 and bev[evt.index[1], evt.index[0]][1] == 0 and bev[evt.index[1], evt.index[0]][2] == 255:
            labels_allowed = labels_allowed_seg
        else:
            labels_allowed = labels_allowed_hdmap
        print(labels_allowed)
        input_point = np.array([[evt.index[0], evt.index[1]]])
        input_label = np.array([1])

        print(lidardata.shape)
        target = np.array([(evt.index[1] - 100)/2, (evt.index[0] - 100)/2, 0])
        pts = []
        for pt in lidardata.T:
            if pt[-1] in labels_allowed:
                pts.append(pt)
        print("final shape lidar: ", np.array(pts).shape)

        arr = np.array(kClosest(np.array(pts), target, 1))
    
        # CAM_BACK
        translation = np.array([-0.02832603, -0.00345137, -1.57910346])
        rotation = np.array([[ 0.00242171,  0.99998907, -0.00400023], [-0.01675361, -0.00395911, -0.99985181], [-0.99985672,  0.00248837,  0.01674384]])
        camera_intrinsic = np.array([[809.2209905677063, 0.0, 829.2196003259838], [0.0, 809.2209905677063, 481.77842384512485], [0.0, 0.0, 1.0]])

        cam_imgs = [cam_left, cam_front, cam_right, cam_rear_left, cam_rear, cam_rear_right]
        cam_keys = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        calibration_data = {
            'CAM_FRONT_LEFT':{
                'translation': [-1.57525595, -0.50051938, -1.50696033],
                'rotation': [[ 0.82254604, -0.56868433, -0.00401771], [ 0.00647832,  0.01643407, -0.99984396], [ 0.56866162,  0.82239167,  0.01720189]],
                'camera_intrinsic': [[1257.8625342125129, 0.0, 827.2410631095686], [0.0, 1257.8625342125129, 450.915498205774], [0.0, 0.0, 1.0]]
            },
            'CAM_FRONT':{
                'translation': [-1.72200568, -0.00475453, -1.49491292],
                'rotation': [[ 0.01026021, -0.99987258, -0.01222952], [ 0.00843345,  0.01231626, -0.99988859], [ 0.9999118 ,  0.01015593,  0.00855874]],
                'camera_intrinsic': [[1252.8131021185304, 0.0, 826.588114781398], [0.0, 1252.8131021185304, 469.9846626224581], [0.0, 0.0, 1.0]]
            },
            'CAM_FRONT_RIGHT':{
                'translation': [-1.58082566,  0.49907871, -1.51749368],
                'rotation': [[-0.84397973, -0.53614138, -0.01583178], [ 0.01645551,  0.00362107, -0.99985804], [ 0.5361226 , -0.84412044,  0.00576637]],
                'camera_intrinsic': [[1256.7485116440405, 0.0, 817.7887570959712], [0.0, 1256.7485116440403, 451.9541780095127], [0.0, 0.0, 1.0]]
            },
            'CAM_BACK_LEFT':{
                'translation': [-1.035691  , -0.48479503, -1.59097015],
                'rotation': [[ 0.94776036,  0.31896113,  0.00375564], [ 0.00866572, -0.0139763 , -0.99986478], [-0.31886551,  0.94766474, -0.01601021]],
                'camera_intrinsic': [[1256.7414812095406, 0.0, 792.1125740759628], [0.0, 1256.7414812095406, 492.7757465151356], [0.0, 0.0, 1.0]]
            },
            'CAM_BACK':{
                'translation': [-0.02832603, -0.00345137, -1.57910346],
                'rotation': [[ 0.00242171,  0.99998907, -0.00400023], [-0.01675361, -0.00395911, -0.99985181], [-0.99985672,  0.00248837,  0.01674384]],
                'camera_intrinsic': [[809.2209905677063, 0.0, 829.2196003259838], [0.0, 809.2209905677063, 481.77842384512485], [0.0, 0.0, 1.0]]
            },
            'CAM_BACK_RIGHT':{
                'translation': [-1.0148781 ,  0.48056822, -1.56239545],
                'rotation': [[-0.93477554,  0.35507456, -0.01080503], [ 0.01587584,  0.0113705 , -0.99980932], [-0.35488399, -0.93476883, -0.01626597]],
                'camera_intrinsic': [[1259.5137405846733, 0.0, 807.2529053838625], [0.0, 1259.5137405846733, 501.19579884916527], [0.0, 0.0, 1.0]]
            }
        }

        min_dist = 1.0
        flag = 0
        cam_img = None
        for cam_ind, cam_key in enumerate(cam_keys):
            barr = np.copy(arr)
            ppc = LidarPointCloud(barr.T);
            ppc.translate(np.array(calibration_data[cam_key]['translation']))
            ppc.rotate(np.array(calibration_data[cam_key]['rotation']))
            ddepths_img = ppc.points[2, :]
            points = view_points(ppc.points[:3, :], np.array(calibration_data[cam_key]['camera_intrinsic']), normalize=True)
            print(points)
            mask = np.ones(ddepths_img.shape[0], dtype=bool)
            mask = np.logical_and(mask, ddepths_img > min_dist)
            mask = np.logical_and(mask, points[0, :] > 1)
            mask = np.logical_and(mask, points[0, :] < 1600 - 1)
            mask = np.logical_and(mask, points[1, :] > 1)
            mask = np.logical_and(mask, points[1, :] < 900 - 1)

            print(mask)
            if mask.sum() > 0:
                # found
                flag = 1
                points = points[:, mask]
                ddepths_img = ddepths_img[mask]
                cam_img = cam_imgs[cam_ind]
                print(points.shape, " points")
                break

        if flag == 0:
            # no point able to back-project, just use front cam
            print("AAAAAAAAA no point found")
            cam_img = cam_front

        predictor.set_image(cam_img)
        input_point = np.array(points.astype(np.int32).T[:, :2])
        input_label = np.array([1 for i in range(len(points[0]))])
        print(input_point, input_label)
        masks, scores, logits = predictor.predict(point_coords=input_point,point_labels=input_label,multimask_output=True)
        print(masks.shape, " masks")
        idxs = (masks[-1].astype(np.uint8) * 200) > 0
        img_copy = np.copy(cam_img)
        img_copy[~idxs] = 0
        img_copy = crop_around_bounding_box(img_copy)
        return img_copy

    bev_img.select(get_select_coords, [cam_left, cam_front, cam_right, cam_rear_left, cam_rear, cam_rear_right, bev_img, lidar_data, tolerance], image)
    
    upload_button.click(upload_img, [image, text_input], [image_tensor, image, text_input, upload_button])
    
    text_input.submit(gradio_ask, [text_input, img, chatbot], [img, streamer, stopping_criteria, input_ids, chatbot]).then(
        gradio_answer, [input_ids, streamer, stopping_criteria, image_tensor, chatbot], [chatbot]
    )
    clear.click(gradio_reset, [img], [chatbot, image, text_input, upload_button], queue=False)

demo.launch(share=True, enable_queue=True)
