# Talk2BEV
Code for:

1. Click2Chat interface

2. JSON generation

## Installation
Please run the following commands
### Setup Talk2BEV

```
git clone https://github.com/llm-bev/talk2bev
```

### Setup LLava

```
git clone https://github.com/haotian-liu/LLaVA parent-folder
mv parent-folder/llava ./
rm -rf parent-folder
```

### Setup LLava
First, you need to clone the repo - 

```
git clone https://github.com/haotian-liu/LLaVA parent-folder
mv parent-folder/llava ./
rm -rf parent-folder
```
Please download the preprocessed weights for [vicuna-13b](https://huggingface.co/liuhaotian/llava-v1-0719-336px-lora-vicuna-13b-v1.3)

### Setup MiniGPT-4 (optional)
```
git clone https://github.com/Vision-CAIR/MiniGPT-4 parent-folder
mv parent-folder/minigpt4 ./
rm -rf parent-folder
```
Please download the preprocessed weights for Vicuna. After downloading the weights, you change the following line in `minigpt4/configs/models/minigpt4.yaml`.
```
16: llama_model: "path-to-llama-preprocessed-weights"
```
Please download the minigpt4 weights [here](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view) and change the link in `eval_configs/minigpt4_eval.yaml`:
```
11: ckpt: 'path-to-prerained_minigpt4_7b-weights'
```

### Setup FastSAM

```
git clone https://github.com/CASIA-IVA-Lab/FastSAM parent-folder
mv parent-folder/FastSAM/fastsam ./
rm -rf parent-folder
```
Download the weights from [here](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view)

### Install SAM (optional)
```
pip3 install segment-anything
```
Download the sam weights from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

## Usage (for Click2Chat Interface)
If using LLaVa
```
python click2chat_llava.py --sam-checkpoint <path-to-sam-checkpoint> --conv-mode <conversion mode, default is llava v1> --model-path <path-llava-model> --gpu-id <gpu num>
```

If using MiniGPT-4
```
python click2chat_minigpt4.py --sam-checkpoint <path-to-sam-checkpoint> --conv-mode <conversion mode, default is llava v1> --model-path <path-llava-model> --gpu-id <gpu num>
```
