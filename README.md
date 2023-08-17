# unsafe-diffusion

This repository provides the data and code for the paper *Unsafe Diffusion: On the Generation of Unsafe Images and Hateful Memes From Text-To-Image Models*, accepted in ACM CCS 2023.

Paper: https://arxiv.org/pdf/2305.13873.pdf

## Unsafe Image Generation


### 1. Collecting Prompts

We use three harmful prompt datasets and one harmless prompt dataset. Request the prompt datasets here: https://zenodo.org/record/8255664

- 4chan prompts (harmful)
- Lexica prompts (harmful)
- Template prompts (harmful)
- COCO prompts (harmless)

### 2. Generating Images

We use four open-sourced Text-to-Image models:

- Stable Diffusion: https://github.com/CompVis/stable-diffusion
- Latent Diffusion: https://github.com/CompVis/latent-diffusion
- DALLE-2 demo: https://github.com/lucidrains/DALLE2-pytorch
- DALLE-mini: https://github.com/borisdayma/dalle-mini

### 3. Unsafe Image Classification

We labeled 800 generated images. Request the image dataset here: https://zenodo.org/record/8255664

**Prerequisite**

```pip install -r requirements.txt```

**Train the Multi-headed Safety Classifier**

```
python train.py
  --images_dir ./data/images \
  --labels_dir ./data/labels.xlsx \
  --output_dir ./checkpoints/multi-headed\
```

**Evaluate the Classifier and Other Baselines**

```
python evaluate.py
  --images_dir ./data/images \
  --labels_dir ./data/labels.xlsx \
  --checkpoints_dir ./checkpoints
```

**Directly Use the Classifier to Detect Unsafe Images**

```
python inference.py
  --images_dir ./data/images \
  --output_dir ./results
```


## Hateful Meme Generation

We employ three image editing techniques on top of Stable Diffusion:

- DreamBooth: https://github.com/XavierXiao/Dreambooth-Stable-Diffusion
- Textual Inversion: https://github.com/rinongal/textual_inversion
- SDEdit: https://github.com/CompVis/stable-diffusion

## Reference

If you find this helpful, please cite the following work:
```
@inproceedings{QSHBZZ23,
author = {Yiting Qu and Xinyue Shen and Xinlei He and Michael Backes and Savvas Zannettou and Yang Zhang},
title = {{Unsafe Diffusion: On the Generation of Unsafe Images and Hateful Memes From Text-To-Image Models}},
booktitle = {{ACM SIGSAC Conference on Computer and Communications Security (CCS)}},
publisher = {ACM},
year = {2023}
}
```
