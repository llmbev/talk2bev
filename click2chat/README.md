# Talk2BEV Click2Chat Interface

Code for:

1. Click2Chat interface

2. JSON generation

## Installation
Please run the following commands
### Setup Talk2BEV

```
git clone https://github.com/llm-bev/talk2bev
```

## Usage (for Click2Chat Interface)
If using LLaVa
```
python click2chat/click2chat_llava.py --sam-checkpoint <path-to-sam-checkpoint> --conv-mode <conversion mode, default is llava v1> --model-path <path-llava-model> --gpu-id <gpu num>
```

If using MiniGPT-4
```
python click2chat/click2chat_minigpt4.py --sam-checkpoint <path-to-sam-checkpoint> --conv-mode <conversion mode, default is llava v1> --model-path <path-llava-model> --gpu-id <gpu num>
```
