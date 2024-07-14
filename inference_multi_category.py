import torch
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import argparse
import os
import sys
import glob
import json
from pathlib import Path
from sklearn import metrics
import random
import open_clip
from baselines import safety_filter_check, Q16_check
import config
import tqdm
from train import BinaryAnnotatedDataset, MHSafetyClassifier

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2022)
random.seed(2022)
np.random.seed(2022)

encode_labels = {"normal": 0, "sexual": 1, "violent": 2,
                 "disturbing": 3, "hateful": 4, "political": 5}
unsafe_contents = list(encode_labels.keys())[1:]


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, valid_extensions=None):
        if valid_extensions is None:
            valid_extensions = ['.jpg', '.jpeg', '.png']
        self.images_dir = images_dir
        self.image_locs = [f for f in os.listdir(images_dir)
                           if os.path.isfile(os.path.join(images_dir, f))
                           and os.path.splitext(f)[1].lower() in valid_extensions]

    def __getitem__(self, idx):
        return os.path.join(self.images_dir, self.image_locs[idx])

    def __len__(self):
        return len(self.image_locs)


def multiheaded_check(loader, checkpoints):
    model_name, pretrained = config.model_name, config.pretrained
    model = MHSafetyClassifier(device, model_name, pretrained)
    model.freeze()
    res = {}
    with torch.no_grad():
        for head in unsafe_contents:
            model.projection_head.load_state_dict(
                torch.load(f"{checkpoints}/{head}.pt", map_location=device))
            model.projection_head.eval()

            res[head] = []
            for batch in loader:
                imgs = batch
                images = [model.preprocess(PIL.Image.open(img_path))
                          for img_path in imgs]
                images = torch.stack(images).to(device)  # [b_s, 3, 224, 224]
                logits = model(images).squeeze()
                preds = (logits.detach().cpu() > 0.5).to(dtype=torch.int64)
                res[head].extend(preds.tolist())

    return res


def main(opt):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    mh_checkpoints = current_directory + "/checkpoints/multi-headed"

    output_dir = opt.output_dir
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    dataset = ImageDataset(images_dir=opt.images_dir)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=50, drop_last=False, shuffle=False)

    res = multiheaded_check(loader=loader, checkpoints=mh_checkpoints)

    final_result = {}
    for i, item in enumerate(dataset):
        for head in res.keys():
            if item not in final_result:
                final_result[item] = {}
            final_result[item][head] = res[head][i]

    json.dump(final_result, open(f"{output_dir}/predictions.json", "w"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images_dir",
        type=str,
        nargs="?",
        default=None,
        help="images folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=None
    )

    opt = parser.parse_args()

    main(opt)
