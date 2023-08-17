import torch
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import argparse, os, sys, glob
import json
from pathlib import Path
from sklearn import metrics
import random
import open_clip
from baselines import safety_filter_check, Q16_check
import config, tqdm
from train import BinaryAnnotatedDataset, MHSafetyClassifier

SEED = 2022
device = torch.device("cuda")
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

encode_labels = {"normal":0, "sexual":1, "violent":2, "disturbing":3, "hateful":4, "political": 5}
unsafe_contents = list(encode_labels.keys())[1:]


def multiheaded_check(loader, checkpoints):
    model_name, pretrained = config.model_name, config.pretrained
    model = MHSafetyClassifier(device, model_name, pretrained)
    model.freeze()
    res = {}
    with torch.no_grad():
        for head in unsafe_contents:
            model.projection_head.load_state_dict(torch.load(f"{checkpoints}/{head}.pt"))
            model.projection_head.eval()
            
            res[head] = []
            for batch in loader:
                imgs, _ = batch
                images = [model.preprocess(PIL.Image.open(img_path)) for img_path in imgs]
                images = torch.stack(images).to(device) # [b_s, 3, 224, 224]
                logits = model(images).squeeze()
                preds = (logits.detach().cpu()>0.5).to(dtype=torch.int64)
                res[head].extend(preds.tolist())
                
    return res

def eval(opt, detector, split="test"):
    
    q16_checkpoint = f"{opt.checkpoints_dir}/q16/prompts.p"
    q16_checkpoint_finetuned = f"{opt.checkpoints_dir}/finetuned_q16/prompts.pt"
    mh_checkpoints = f"{opt.checkpoints_dir}/multi-headed"

    dataset = BinaryAnnotatedDataset(images_dir=opt.images_dir, labels_dir=opt.labels_dir, split=split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=50, drop_last=False, shuffle=False)    
    
    if detector == "safety_checker":
        res = safety_filter_check(loader=loader)
    elif detector == "q16":
        res = Q16_check(loader=loader, checkpoint=q16_checkpoint)
    elif detector == "finetuned_q16":
        res = Q16_check(loader=loader, checkpoint=q16_checkpoint_finetuned)
    elif detector == "multi-headed":
        res = multiheaded_check(loader=loader, checkpoints=mh_checkpoints)

    ground_truth = [dataset.__getitem__(idx)[1] for idx in range(len(dataset))]
    if detector == "multi-headed":
        _preds = []
        for head in unsafe_contents:
            _preds.append(res[head])
        _preds = np.array(_preds)
        preds = np.sum(_preds, axis=0)
        preds = np.int16(preds>0)
    else:
        preds = res

    
    accuracy = metrics.accuracy_score(ground_truth, preds)
    precision = metrics.precision_score(ground_truth, preds)
    recall = metrics.recall_score(ground_truth, preds)
    f1_score = metrics.f1_score(ground_truth, preds)
    print("accuracy, precision, recall, f1_score")
    print(f"{detector}, {accuracy:.2f}, {precision:.2f}, {recall:.2f}, {f1_score:.2f}")


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--images_dir",
        type=str,
        nargs="?",
        default=None,
        help="images folder"
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        nargs="?",
        default=None,
        help="the directory saved prompts"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=None
    )

    opt = parser.parse_args()
    
    for detector in ["safety_checker", "q16", "finetuned_q16", "multi-headed"]:
        eval(opt, detector)
