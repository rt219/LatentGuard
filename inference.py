import numpy as np
import torch
import random
import time
from sklearn.metrics import roc_auc_score
import sys
from utils import *
import argparse

parser = argparse.ArgumentParser(description="Parameters of inference LatentGuard model")
parser.add_argument('--file_path', type=str, default='unsafe_sample.txt', help="Path to the unsafe file")
parser.add_argument('--threshold', type=float, default=9.0131, help="Threshold value")
args = parser.parse_args()

# Config
unsafe_harm = read_unsafe_file(args.file_path)
threshold = args.threshold

# Init & load model
device = 'cuda:0'
num_heads=16; head_dim=32; out_dim=128
model = EmbeddingMappingLayer(num_heads, head_dim, out_dim).to(device)
model.load_state_dict(torch.load("model_parameters.pth"))

def eval_on_unsafe(is_train_concepts = True,
    candidate_raw_data_list = unsafe_harm):

    model.eval()
    pred_list = []

    from collections import defaultdict

    target_concept_set =  train_concepts if is_train_concepts else test_concepts
    target_concept_set = list(target_concept_set)

    print('Preparing concept embeddings... it may take seconds...')
    concept_embs = [wrapClip.get_emb(concept).to(device) for concept in target_concept_set]
    print('Concept embeddings prepared.')
    
    all_concept_emb = torch.cat(concept_embs, dim=0).to(device)
    all_concept_emb = all_concept_emb[:, 0, :]

    print('Number of prompts to be evaluated: ', len(candidate_raw_data_list))
    touse_list = candidate_raw_data_list

    selected_items = touse_list

    print('Predicting...')
    info = []
    for _i, prompt_data in enumerate(selected_items):
        if _i%100==0:
            print(f'process {_i}')
        cur_concept = None
        prompt = prompt_data
        
        prompt_emb = wrapClip.get_emb(prompt).to(device)

        with torch.no_grad():
            prompt_emb = prompt_emb.to(device)
            repeated_prompt_emb = prompt_emb.repeat(len(target_concept_set), 1, 1)
            output = model(repeated_prompt_emb.to(device), all_concept_emb.to(device))
            dot_product = forward_contra_model(model, output)
            
            predicted_maxv = dot_product.max(0)[0].cpu().numpy()
            pred_list.append(predicted_maxv)
    return pred_list

def run(model, title):
    pred_harm = eval_on_unsafe(is_train_concepts = True,
        candidate_raw_data_list = unsafe_harm)
    pred = np.array(pred_harm)
    pred_labels = (pred >= threshold).astype(int)
    return pred_labels

pred_labels = run(model, 'inference')
print("\n\nInference completed.")
print(f"{len(pred_labels)} prompts predicted")
print("Predicted labels(1 represents harmful):")
print(pred_labels)