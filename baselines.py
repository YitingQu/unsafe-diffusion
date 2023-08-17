import torch
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import argparse, os, sys, glob
import json
from pathlib import Path
import torch.nn as nn
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from sklearn import metrics
import random
import open_clip
import logging
import pickle
import config

device = torch.device("cuda")

#========================================== Stable Diffusion built-in Classifier =====================================

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    # images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

    
def safety_filter_check(loader):

    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id).to(device)
    safety_checker.eval()

    def check_safety(x_image, device): # [1, W, H, C]
        safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt") # pixel_values: [1, 3, 224, 224]
        safety_checker_input = safety_checker_input.to(device)
        x_checked_image, has_nsfw_concept = safety_checker(clip_input=safety_checker_input.pixel_values, images=x_image)
        assert x_checked_image.shape[0] == len(has_nsfw_concept)
        return has_nsfw_concept

    res = []
    for batch in loader:
        imgs, _ = batch
        for img in imgs:
            img = Image.open(img).convert("RGB")
            img_array = np.expand_dims(np.array(img), axis=0)
            has_nsfw_concept = check_safety(img_array, device)[0]
            res.append(int(has_nsfw_concept))
         
    return res

#========================================== Q16 Model =====================================

class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings, model_name, pretrained):
        super(SimClassifier, self).__init__()
        self.clip_model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained)
        self.clip_model.to(torch.float32)

        self.prompts = torch.nn.Parameter(embeddings)
        
    def freeze(self):
        self.clip_model = self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        image_feat = self.clip_model.encode_image(x)
        # Pick the top 5 most similar labels for the image
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_feat @ text_feat.T)
        # values, indices = similarity[0].topk(5)
        return similarity.squeeze()

def initialize_prompts(model, text_prompts, device):
    text = model.preprocess(text_prompts).to(device)
    return model.clip_model.encode_text(text)

def load_prompts(file_path, device):
    if file_path.endswith("p"):
        res = torch.FloatTensor(pickle.load(open(file_path, 'rb'))).to(device)
    elif file_path.endswith("pt"):
        res = torch.load(open(file_path, 'rb')).to(device).to(torch.float32)
    return res


def Q16_check(loader, checkpoint):

    model_name, pretrained = config.model_name, config.pretrained
    soft_prompts = load_prompts(checkpoint, device)
    classifier = SimClassifier(soft_prompts, model_name, pretrained)
    classifier.freeze()
    classifier.to(device)

    res = []
    for batch in loader:
        imgs, _ = batch
        images = [classifier.preprocess(PIL.Image.open(img_path)) for img_path in imgs]
        images = torch.stack(images).to(device) # [b_s, 3, 224, 224]
        y = classifier(images)
        y = torch.argmax(y, dim=1).detach().cpu().numpy()
        res.extend(y.tolist())

    return res
