# Talk2BEV-Dataset from Scratch

## Installation

To generate captions, setup the baselines using the following commands:

### LLava

```bash
git clone https://github.com/haotian-liu/LLaVA parent-folder
mv parent-folder/llava ./
rm -rf parent-folder
```

Please download the preprocessed weights for [vicuna-13b](https://huggingface.co/liuhaotian/llava-v1-0719-336px-lora-vicuna-13b-v1.3)

### MiniGPT-4 (optional)

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4 parent-folder
mv parent-folder/minigpt4 ./
rm -rf parent-folder
```

Please download the preprocessed weights for Vicuna. After downloading the weights, you change the following line in `minigpt4/configs/models/minigpt4.yaml`.

```bash
16: llama_model: "path-to-llama-preprocessed-weights"
```

Please download the minigpt4 weights [here](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view) and change the link in `eval_configs/minigpt4_eval.yaml`:

```bash
11: ckpt: 'path-to-prerained_minigpt4_7b-weights'
```

### FastSAM

```bash
git clone https://github.com/CASIA-IVA-Lab/FastSAM parent-folder
mv parent-folder/FastSAM/fastsam ./
rm -rf parent-folder
```

Download the weights from [here](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view)

### Install SAM (optional)

```bash
pip3 install segment-anything
```

Download the sam weights from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).


## Base

To generate the base, please run the following commands:

```bash
cd data
python3 generate_base.py --data_path <path-to-nuscenes-v1.0-trainval> --save_path <path-to-save> --bev pred/gt
```

## Captioning

To generate the captions for each scene object, please run the following commands:

```bash
python3 generate_captions.py --model <captioning-model> --data_path <path-to-base-folder> --json_name pred/gt --start <start_index> --end <end-index> 
```
