# Talk2BEV: Language-Enhanced Bird's Eye View Maps

[**Project Page**](https://llmbev.github.io/talk2bev/) |
[**ArXiv**]() |
[**Video**](https://www.youtube.com/watch?v=TvMeekCAfRs)

[Vikrant Dewangan](https://vikr-182.github.io/)\* <sup>**1**</sup>,
[Tushar Choudhary](https://tusharc31.github.io/)\* <sup>**1**</sup>,
[Shivam Chandhok](https://scholar.google.com/citations?user=ZER2BeIAAAAJ&hl=en)\* <sup>**2**</sup>,
[Shubham Priyadarshan](https://rudeninja.github.io/) <sup>**1**</sup>,
[Anushka Jain](https://anushkaj1.github.io) <sup>**1**</sup>,
[Arun K. Singh](https://scholar.google.co.in/citations?user=0zgDoIEAAAAJ&hl=en) <sup>**3**</sup>,
[Siddharth Srivastava](https://siddharthsrivastava.github.io/) <sup>**4**</sup>,
[Krishna Murthy Jatavallabhula](https://krrish94.github.io/) $^\dagger$ <sup>**5**</sup>,
[K. Madhava Krishna](https://scholar.google.co.in/citations?user=QDuPGHwAAAAJ&hl=en) $^\dagger$ <sup>**1**</sup>

<sup>**1**</sup> International Institute of Information Technology Hyderabad,
<sup>**2**</sup> University of British Columbia,
<sup>**3**</sup> University of Tartu
<sup>**4**</sup> TensorTour Inc
<sup>**5**</sup> MIT-CSAIL

> *denotes equal contribution,
$^\dagger$ denotes equal advising

> **Under Review at** [ICRA 2024](https://2024.ieee-icra.org/)


https://github.com/llmbev/talk2bev/assets/44745884/251ffffd-2bd7-4838-895a-83939ec5b87f

## Abstract

We introduce Talk2BEV, a large vision- language model (LVLM) interface for bird’s-eye view (BEV) maps commonly used in autonomous driving.

While existing perception systems for autonomous driving scenarios have largely focused on a pre-defined (closed) set of object categories and driving scenarios, Talk2BEV eliminates the need for BEV- specific training, relying instead on performant pre-trained LVLMs. This enables a single system to cater to a variety of autonomous driving tasks encompassing visual and spatial reasoning, predicting the intents of traffic actors, and decision- making based on visual cues.

We extensively evaluate Talk2BEV on a large number of scene understanding tasks that rely on both the ability to interpret freefrom natural language queries, and in grounding these queries to the visual context embedded into the language-enhanced BEV map. To enable further research in LVLMs for autonomous driving scenarios, we develop and release Talk2BEV-Bench, a benchmark encom- passing 1000 human-annotated BEV scenarios, with more than 20,000 questions and ground-truth responses from the NuScenes dataset.

## Data Preparation

Please download the [NuScenes v1.0-trainval](https://www.nuscenes.org/download) dataset. Our dataset consists of 2 parts - Talk2BEV-Base and Talk2BEV-Captions, consisting of base (crops, perspective images, bev area centroids) and crop captions respectively.

### Download links

We provide 2 Links to the Talk2BEV dataset (_Talk2BEV-Mini_ (captions only) and _Talk2BEV-Full_) are provided below. The dataset is hosted on Google Drive. Please download the dataset and extract the files to the `data` folder.

| Name | Base | Captions | Bench | Link |
| --- | --- | --- | --- | --- |
| Talk2BEV-Mini |  &check; | &cross; | &cross; | [link](https://drive.google.com/file/d/1B5Uong8xYGRDkufR33T9sCNyNdRzPxc4/view?usp=sharing) |
| Talk2BEV-Full | &cross; | &cross; | &cross; | _TODO_ |

If you want to generate the dataset from scratch, please follow the process [here](./data/scratch.md). The format for each of the data parts is described in [format](./data/format.md).

## Evaluation

Evaluation on Talk2BEV happens via 2 methods - MCQs (from Talk2BEV-Bench) and Spatial Operators. We use GPT-4 for our evaluation. Please follow the instructions in [GPT-4](https://platform.openai.com/) and initialize the API key and Organization in your os env.

```bash
ORGANIZATION=<your-organization>
API_KEY=<your-api-key>
```

### Evaluating - MCQs

To obtain the accuracy for a MCQs, please run the following command:

```bash
cd evaluation
python eval_mcq.py
```

This will yield the accuracy for the MCQs.

### Evaluating Spatial Operators

To obtain the distance error, IoU for a MCQs, please run the following command:

```bash
cd evaluation
python eval_spops.py
```

## Click2Chat

We also allow free-form conversation with the BEV. Please follow the instructions in [Click2Chat](./click2chat/README.md) to chat with the BEV.

### Talk2BEV-Bench

TO BE RELEASED

## 👉 TODO 

- [x] Spatial operators evaluation pipeline
- [ ] Add links to BEV crops -- Release Talk2BEV-Full
- [ ] Release Talk2BEV-Bench
