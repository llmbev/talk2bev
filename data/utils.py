from PIL import Image
import argparse
from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from llava.conversation import conv_templates

from lavis.models import load_model_and_preprocess

from tqdm import tqdm
import argparse

cam_keys = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

# ========================================
#             InsructBLIP-2 Model Initialization
# ========================================
def init_instructblip2(model_name = "blip2_vicuna_instruct", device="cuda:0"):
    model, vis_processors, _ = load_model_and_preprocess(
        name=model_name,
        model_type="vicuna13b",
        is_eval=True,
        device=device,
    )
    return model, vis_processors
# ========================================

# ========================================
#             BLIP-2 Model Initialization
# ========================================
def init_blip2(model_name = "blip2_vicuna_instruct", device="cuda:0", model_type="vicuna13b"):
    model, vis_processors, _ = load_model_and_preprocess(
        name=model_name,
        model_type=model_type,
        is_eval=True,
        device=device,
    )
    return model, vis_processors
# ========================================


# ========================================
#             MiniGPT4 Initialization
# ========================================
def init_minigp4():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--sam-checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="path to sam weights.")
    parser.add_argument('--model_path', type=str, default="/raid/t1/scratch/vikrant.dewangan/LLaVA/ckpt-old/", help='save path for jsons')
    parser.add_argument('--save_path', type=str, default="/raid/t1/scratch/vikrant.dewangan/datas", help='save path for jsons')
    parser.add_argument('--gpu', type=str, default="cuda:0", help='save path for jsons')
    parser.add_argument('--json_name', type=str, default="answer_pred_both.json", help='save path for jsons')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=100, help='end index')

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    print('Initializing Chat')
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    return chat
# ========================================

def reset_conv(model_name = "llava"):
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    print('reset conv')

    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    return conv

def minigpt4_inference(chat, img_cropped, user_message):
    img_list = []
    chat_state = CONV_VISION.copy()  # Reset chat state to default template
    llm_message = chat.upload_img(Image.fromarray(img_cropped), chat_state, img_list)

    print('Upload done')

    chat.ask(user_message, chat_state)
    llm_message = chat.answer(
            conv=chat_state,
            img_list=img_list,
            # num_beams=num_beams,
            num_beams=1,
            # temperature=temperature,
            temperature=0.7,
            max_new_tokens=300,
            max_length=2000
    )[0]
    return llm_message

def instructblip2_inference(model_instructblip, img_cropped, vis_processors, device="cuda:0", user_message="describe the central object in the scene."):
    image = vis_processors["eval"](Image.fromarray(img_cropped)).unsqueeze(0).to(device)

    samples = {
        "image": image,
        "prompt": user_message,
    }

    output_blip = model_instructblip.generate(
        samples,
        length_penalty=float(1),
        repetition_penalty=float(1),
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.2,
        use_nucleus_sampling=False,
    )

    return output_blip[0]