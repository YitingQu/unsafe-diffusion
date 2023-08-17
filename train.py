import torch
import numpy as np
import pandas as pd
import PIL
import argparse, os, sys, glob
import json
from pathlib import Path
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn import functional as F
from sklearn import metrics
import random
import open_clip
import logging
import config, tqdm
from collections import Counter
import shutil
from sklearn import model_selection

device = torch.device("cuda")
SEED = 2022
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

encode_labels = {"normal":0, "sexual":1, "violent":2, "disturbing":3, "hateful":4, "political": 5}
unsafe_contents = list(encode_labels.keys())[1:]

class BinaryAnnotatedDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, split="train", head=None, train_test_split=0.4):
        
        labels_df = pd.read_excel(labels_dir)
        images, labels = [], []
        for i in labels_df.index:
            images.append(f"{images_dir}/{i}.png")
            label = labels_df.loc[i, "final_label"]
            
            raw_labels = []
            for annotator in ["rater_0", "rater_1", "rater_2"]:
                _label = labels_df.loc[i, annotator]
                _label = [int(l) for l in str(_label).split(",")]
                raw_labels.extend(_label)
            label_collection = Counter(raw_labels).most_common()
            label_collection_dict = {}
            for l, n in label_collection:
                label_collection_dict[l] = n
            if head:
                target_label = encode_labels[head]
                try:
                    if int(label_collection_dict[target_label]) >= 2:
                        label = 1
                except:
                    label = 0

            labels.append(label)

        images_train, images_test, labels_train, labels_test = model_selection.train_test_split(images, labels, \
                                                                                            test_size=train_test_split, 
                                                                                            shuffle=True, 
                                                                                        random_state=1)     
        if split == "train":
            self.images = images_train
            self.labels = labels_train
        elif split == "test":
            self.images = images_test
            self.labels = labels_test
        
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]   
    
    def __len__(self):
        return len(self.images)
         
    def weights(self):
        count = Counter(self.labels)
        class_count = np.array([count[0], count[1]])
        weight = 1.0/class_count
        weights = np.array([weight[0] if la==0 else weight[1] for la in self.labels])
        return weights


class MHSafetyClassifier(torch.nn.Module):
    def __init__(self, device, model_name, pretrained):
        super(MHSafetyClassifier, self).__init__()
        self.clip_model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained)
        self.clip_model.to(device)
        self.projection_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(384),
            nn.Linear(384, 1)
            ).to(device)

    def freeze(self):
        self.clip_model = self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.clip_model.encode_image(x).type(torch.float32)
        x = self.projection_head(x)
        out = nn.Sigmoid()(x)
        return out
    
def train(opt, record=True):

    EPOCH = config.epoch
    LR = config.learning_rate
    BATCH_SIZE = config.batch_size

    model_name, pretrained = config.model_name, config.pretrained
    output_dir = opt.output_dir
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    for head in unsafe_contents:
        if record:
            logging.getLogger('').handlers = []
            logging.basicConfig(level=logging.INFO, filename=f"{output_dir}/{head}.log")
        
        
        trainset = BinaryAnnotatedDataset(images_dir=opt.images_dir, labels_dir=opt.labels_dir, split="train", head=head)
        sampler = WeightedRandomSampler(trainset.weights(), num_samples=trainset.weights().shape[0], replacement=True)
        testset = BinaryAnnotatedDataset(images_dir=opt.images_dir, labels_dir=opt.labels_dir, split="test", head=head)

        train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, drop_last=True, sampler=sampler)
        test_loader = DataLoader(testset, batch_size=20, drop_last=False)
        
        model = MHSafetyClassifier(device, model_name, pretrained)
        model.freeze() # freeze the backbone
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.projection_head.parameters(), lr=LR)
        
        best_score = 0.0
        for epoch in range(EPOCH):
            model.projection_head.train()
            total_loss = 0.0 
            ground_truth, prediction = [], []
            for idx, (imgs, labels) in enumerate(train_loader):
                labels = labels.to(device)
                labels = labels.type(torch.float32)
                images = [model.preprocess(PIL.Image.open(img_path)) for img_path in imgs]
                images = torch.stack(images).to(device) # [b_s, 3, 224, 224]
                logits = model(images).squeeze()
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                preds = (logits.detach().cpu()>0.5).to(dtype=torch.int64)
                ground_truth.append(labels.detach().cpu())
                prediction.append(preds)
                # print(loss)
            avg_loss = total_loss/(idx+1)
            ground_truth = torch.hstack(ground_truth)
            prediction = torch.hstack(prediction)
            accuracy = metrics.accuracy_score(ground_truth.numpy(), prediction.numpy())
            if record:
                logging.info(f"[epoch] {epoch} [train accuracy] {accuracy} [loss] {loss}")

            #================= eval =================
            test_ground_truth, test_prediction = [], []
            model.projection_head.eval()
            for idx, (imgs, labels) in enumerate(test_loader):
                labels = labels.to(device)
                images = [model.preprocess(PIL.Image.open(img_path)) for img_path in imgs]
                images = torch.stack(images).to(device) # [b_s, 3, 224, 224]
                logits = model(images).squeeze()
                preds = (logits.detach().cpu()>0.5).to(dtype=torch.int64)
                test_ground_truth.append(labels.detach().cpu())
                test_prediction.append(preds)
            
            test_ground_truth = torch.hstack(test_ground_truth)
            test_prediction = torch.hstack(test_prediction)
            accuracy = metrics.accuracy_score(test_ground_truth.numpy(), test_prediction.numpy())
            precision = metrics.precision_score(test_ground_truth.numpy(), test_prediction.numpy())
            recall = metrics.recall_score(test_ground_truth.numpy(), test_prediction.numpy())
            f1_score = metrics.f1_score(test_ground_truth.numpy(), test_prediction.numpy())

            print(f"{head} test-performance: [accuracy] {accuracy} [precision] {precision} [recall] {recall}  [f1_score]  {f1_score} \n")
            
            if accuracy > best_score:
                best_score = accuracy
                torch.save(model.projection_head.state_dict(), f"{output_dir}/{head}.pt")
                print(f"best model - {head} {epoch} {best_score}")

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
        "--output_dir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=None
    )
    
    opt = parser.parse_args()
    
    train(opt, record=True)