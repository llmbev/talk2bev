# Talk2BEV

[**Project Page**](https://llmbev.github.io/talk2bev/) |
[**Paper**](https://llmbev.github.io/talk2bev/assets/pdf/talk2bev.pdf) |
[**ArXiv**]() |
[**Video**](https://www.youtube.com/watch?v=TMht-8SGJ0I)

[Vikrant Dewangan](https://vikr-182.github.io/)\*,
[Tushar Choudhary](https://tusharc31.github.io/)\*,
[Shivam Chandhok](https://scholar.google.com/citations?user=ZER2BeIAAAAJ&hl=en)\*,
[Shubham Priyadarshan](https://rudeninja.github.io/),
[Anushka Jain](https://anushkaj1.github.io),
[Arun K. Singh](https://scholar.google.co.in/citations?user=0zgDoIEAAAAJ&hl=en),
[Siddharth Srivastava](https://siddharthsrivastava.github.io/),
[Krishna Murthy Jatavallabhula](https://krrish94.github.io/)\*,
[K. Madhava Krishna](https://scholar.google.co.in/citations?user=QDuPGHwAAAAJ&hl=en),

![Splash Figure](./docs/static/images/talk2bev_teaser-1.png)

## Abstract

We introduce Talk2BEV, a large vision- language model (LVLM) interface for bird’s-eye view (BEV) maps commonly used in autonomous driving.

While existing perception systems for autonomous driving scenarios have largely focused on a pre-defined (closed) set of object categories and driving scenarios, Talk2BEV eliminates the need for BEV- specific training, relying instead on performant pre-trained LVLMs. This enables a single system to cater to a variety of autonomous driving tasks encompassing visual and spatial reasoning, predicting the intents of traffic actors, and decision- making based on visual cues.

We extensively evaluate Talk2BEV on a large number of scene understanding tasks that rely on both the ability to interpret freefrom natural language queries, and in grounding these queries to the visual context embedded into the language-enhanced BEV map. To enable further research in LVLMs for autonomous driving scenarios, we develop and release Talk2BEV-Bench, a benchmark encom- passing 1000 human-annotated BEV scenarios, with more than 20,000 questions and ground-truth responses from the NuScenes dataset.

## Data Preparation

Please download the [NuScenes v1.0-trainval](https://www.nuscenes.org/download) dataset. Our dataset consists of 2 parts - Talk2BEV-Base and Talk2BEV-Captions, consisting of base (crops, perspective images, bev area centroids) and crop captions respectively.

### Download links

Links to the Talk2BEV dataset (_Talk2BEV-Base_ and _Talk2BEV-Captions_) are provided below. The dataset is hosted on Google Drive. Please download the dataset and extract the files to the `data` folder.

| Data Parts | Link |
| --- | --- |
| Talk2BEV-Base | [link]() |
| Talk2BEV-Captions | [link]() |

If you want to generate the dataset from scratch, please follow the process [here](./data/scratch.md). The format for each of the data parts is described in [format](./data/format.md).

## Evaluation

Evaluation on Talk2BEV happens via 2 methods - MCQs (from Talk2BEV-Bench) and Spatial Operators. We use GPT-4 for our evaluation. Please follow the instructions in [GPT-4](https://platform.openai.com/) and initialize the API key and Organization in your os env.

```bash
ORGANIZATION=<your-organization>
API_KEY=<your-api-key>
```

### Talk2BEV-Bench

TO BE RELEASED

### Evaluating - MCQs

To obtain the accuracy for a MCQs, please run the following command:

```bash
python eval_mcq.py
```

This will yield the accuracy for the MCQs.

### Evaluating Spatial Operators

TO BE RELEASED

## Click2Chat

We also allow free-form conversation with the BEV. Please follow the instructions in [Click2Chat](./click2chat/README.md) to chat with the BEV.

## TODO

```
[ ] Add links to BEV crops, captions
[ ] Spatial operators evaluation pipeline
[ ] Release Talk2BEV-Bench
```