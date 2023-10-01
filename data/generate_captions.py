import os
import numpy as np
import json
from tqdm import tqdm
import argparse
from utils import init_minigp4, minigpt4_inference, init_blip2, init_instructblip2, instructblip2_inference

cam_keys = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

def run_minigpt4(index_start, index_end, gpu, model_path, save_path = "/raid/t1/scratch/vikrant.dewangan/datas", json_name="answer_gt.json"):
    json_list = os.listdir(save_path)
    num_to_json = {}
    for ind, objp in enumerate(json_list):
        if os.path.exists(os.path.join(save_path, objp, "answer_gt.json")):
            num_to_json[int(objp.split("_")[1])] = objp

    chat = init_minigp4()

    for ind in tqdm(range(index_start, index_end)):
        objs = json.load(open(os.path.join(save_path, num_to_json[ind], json_name)))
        imc = {}
        print(os.path.join(save_path, num_to_json[ind]))
        for j in range(6):
            img = np.load(os.path.join(save_path, num_to_json[ind], "crop_imgs", f"{j + 1}_cimg.npy"))
            user_message = "Describe this image."
            answer = minigpt4_inference(chat, img, user_message)
            imc[cam_keys[j]] = {}
            imc[cam_keys[j]]["description"] = answer

            user_message = "Is there anything unusual about this image?"
            answer = minigpt4_inference(chat, img, user_message)
            imc[cam_keys[j]]["unusual"] = answer

            user_message = "Describe the weather in this image. Is it day or night?"
            answer = minigpt4_inference(chat, img, user_message)
            imc[cam_keys[j]]["weather"] = answer
        
        print("Image Captioning done")
        for indd, obj in enumerate(objs):
            if "pred" in json_name:
                img = np.load(os.path.join(save_path, num_to_json[ind], "crops", f"{indd + 1}_matched_img_pred.npy"))
            else:
                img = np.load(os.path.join(save_path, num_to_json[ind], "crops", f"{indd + 1}_matched_img.npy"))
            user_message = "Describe the central object in this image."
            answer = minigpt4_inference(chat, img, user_message)
            objs[indd]["minigpt4_crop_brief"] = answer

            user_message = "Is the object's indicator on? If yes, what direction does it want to turn. Please answer carefully."
            answer = minigpt4_inference(chat, img, user_message)
            objs[indd]["minigpt4_crop_lights1"] = answer

            user_message = "In this image, are the object's rear lights closer  to the viewer or forward lights? Please answer in one word - forward/rear."
            answer = minigpt4_inference(chat, img, user_message)
            objs[indd]["minigpt4_crop_lights2"] = answer.lower()

            user_message = "Is there any text written on the object? Please look carefully, and describe it."
            answer = minigpt4_inference(chat, img, user_message)
            objs[indd]["minigpt4_crop_text"] = answer

            objs[indd]["minigpt4_bg_description"] = imc[objs[indd]["matched_cam"]]["description"]
            objs[indd]["minigpt4_bg_unusual"] = imc[objs[indd]["matched_cam"]]["unusual"]
            objs[indd]["minigpt4_bg_weather"] = imc[objs[indd]["matched_cam"]]["weather"]

        print("Object Captioning Done")
        save_path2 = "/raid/t1/scratch/vikrant.dewangan/datas5"
        os.makedirs(os.path.join(save_path2, num_to_json[ind]), exist_ok=True)
        with open(os.path.join(save_path2, num_to_json[ind], json_name), "w") as f:
            json.dump(objs, f, indent=4)
        print("DONE SAVED", os.path.join(save_path2, num_to_json[ind]))

def run_instructblip2(index_start, index_end, gpu, model_path, save_path = "/raid/t1/scratch/vikrant.dewangan/datas", json_name="answer_gt.json"):
    json_list = os.listdir(save_path)
    num_to_json = {}
    for ind, objp in enumerate(json_list):
        if os.path.exists(os.path.join(save_path, objp, "answer_gt.json")):
            num_to_json[int(objp.split("_")[1])] = objp

    model_instructblip, vis_processors = init_instructblip2(model_name = "blip2_vicuna_instruct", device="cuda:0")

    for ind in tqdm(range(index_start, index_end)):
        objs = json.load(open(os.path.join(save_path, num_to_json[ind], json_name)))
        imc = {}
        for j in range(6):
            img = np.load(os.path.join(save_path, num_to_json[ind], "crop_imgs", f"{j + 1}_cimg.npy"))
            user_message = "Describe this image."
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            imc[cam_keys[j]] = {}
            imc[cam_keys[j]]["description"] = answer

            user_message = "Is there anything unusual about this image?"
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            imc[cam_keys[j]]["unusual"] = answer

            user_message = "Describe the weather in this image. Is it day or night?"
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            imc[cam_keys[j]]["weather"] = answer
        
        print("Image Captioning done")
        for indd, obj in enumerate(objs):
            if "pred" in json_name:
                img = np.load(os.path.join(save_path, num_to_json[ind], "crops", f"{indd + 1}_matched_img_pred.npy"))
            else:
                img = np.load(os.path.join(save_path, num_to_json[ind], "crops", f"{indd + 1}_matched_img.npy"))
            user_message = "Describe the central object in this image."
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            objs[indd]["instructblip2_crop_brief"] = answer

            user_message = "Is the object's indicator on? If yes, what direction does it want to turn. Please answer carefully."
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            objs[indd]["instructblip2_crop_lights1"] = answer

            user_message = "In this image, are the object's rear lights closer  to the viewer or forward lights? Please answer in one word - forward/rear."
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            objs[indd]["instructblip2_crop_lights2"] = answer.lower()

            user_message = "Is there any text written on the object? Please look carefully, and describe it."
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            objs[indd]["instructblip2_crop_text"] = answer

            objs[indd]["instructblip2_bg_description"] = imc[objs[indd]["matched_cam"]]["description"]
            objs[indd]["instructblip2_bg_unusual"] = imc[objs[indd]["matched_cam"]]["unusual"]
            objs[indd]["instructblip2_bg_weather"] = imc[objs[indd]["matched_cam"]]["weather"]

        print("Object Captioning Done")
        save_path2 = "/raid/t1/scratch/vikrant.dewangan/datas4"
        os.makedirs(os.path.join(save_path2, num_to_json[ind]), exist_ok=True)
        with open(os.path.join(save_path2, num_to_json[ind], json_name), "w") as f:
            json.dump(objs, f, indent=4)
        print("DONE SAVED", os.path.join(save_path2, num_to_json[ind]))

def run_blip2(index_start, index_end, gpu, model_path, save_path = "/raid/t1/scratch/vikrant.dewangan/datas", json_name="answer_gt.json"):
    json_list = os.listdir(save_path)
    num_to_json = {}
    for ind, objp in enumerate(json_list):
        if os.path.exists(os.path.join(save_path, objp, "answer_gt.json")):
            num_to_json[int(objp.split("_")[1])] = objp

    model_instructblip, vis_processors = init_blip2(model_name = "blip2_t5", device="cuda:0", model_type="pretrain_flant5xxl")

    for ind in tqdm(range(index_start, index_end)):
        objs = json.load(open(os.path.join(save_path, num_to_json[ind], json_name)))
        imc = {}
        for j in range(6):
            img = np.load(os.path.join(save_path, num_to_json[ind], "crop_imgs", f"{j + 1}_cimg.npy"))
            user_message = "Describe this image."
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            imc[cam_keys[j]] = {}
            imc[cam_keys[j]]["description"] = answer

            user_message = "Is there anything unusual about this image?"
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            imc[cam_keys[j]]["unusual"] = answer

            user_message = "Describe the weather in this image. Is it day or night?"
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            imc[cam_keys[j]]["weather"] = answer
        
        print("Image Captioning done")
        for indd, obj in enumerate(objs):
            if "pred" in json_name:
                img = np.load(os.path.join(save_path, num_to_json[ind], "crops", f"{indd + 1}_matched_img_pred.npy"))
            else:
                img = np.load(os.path.join(save_path, num_to_json[ind], "crops", f"{indd + 1}_matched_img.npy"))
            user_message = "Describe the central object in this image."
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            objs[indd]["blip2_crop_brief"] = answer

            user_message = "Is the object's indicator on? If yes, what direction does it want to turn. Please answer carefully."
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            objs[indd]["blip2_crop_lights1"] = answer

            user_message = "In this image, are the object's rear lights closer  to the viewer or forward lights? Please answer in one word - forward/rear."
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            objs[indd]["blip2_crop_lights2"] = answer.lower()

            user_message = "Is there any text written on the object? Please look carefully, and describe it."
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            objs[indd]["blip2_crop_text"] = answer

            objs[indd]["blip2_bg_description"] = imc[objs[indd]["matched_cam"]]["description"]
            objs[indd]["blip2_bg_unusual"] = imc[objs[indd]["matched_cam"]]["unusual"]
            objs[indd]["blip2_bg_weather"] = imc[objs[indd]["matched_cam"]]["weather"]

        save_path2 = "/raid/t1/scratch/vikrant.dewangan/datas7"
        os.makedirs(os.path.join(save_path2, num_to_json[ind]), exist_ok=True)
        with open(os.path.join(save_path2, num_to_json[ind], json_name), "w") as f:
            json.dump(objs, f, indent=4)
        print("DONE SAVED", os.path.join(save_path2, num_to_json[ind]))

def main():
    parser = argparse.ArgumentParser(description='Generate captions.')    
    parser.add_argument('--model_path', type=str, default="/raid/t1/scratch/vikrant.dewangan/LLaVA/ckpt-old/", help='save path for jsons')
    parser.add_argument('--data_path', type=str, default="/raid/t1/scratch/vikrant.dewangan/datas", help='save path for jsons')
    parser.add_argument('--gpu', type=str, default="cuda:0", help='gpu number')
    parser.add_argument('--bev', type=str, default="pred", help='pred or gt')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=100, help='end index')
    parser.add_argument('--model', type=str, required=True, help='model name, use minigpt4 or instructblip2 or blip2')

    args = parser.parse_args()
    args.save_path = args.data_path
    if args.bev == "pred":
        args.json_name = "answer_pred_both.json"
    if args.bev == "gt":
        args.json_name = "answer_gt.json"

    if args.model == "minigpt4":
        run_minigpt4(args.start, args.end, args.gpu, args.model_path, args.save_path, json_name=args.json_name)
    elif args.model == "instructblip2":
        run_instructblip2(args.start, args.end, args.gpu, args.model_path, args.save_path, json_name=args.json_name)
    elif args.model == "blip2":
        run_blip2(args.start, args.end, args.gpu, args.model_path, args.save_path, json_name=args.json_name)

if __name__ == "__main__":
    main()