import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
import random
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import time
from sklearn.metrics import roc_auc_score
import sys

from utils import *

from tqdm import tqdm

def do_cache(clip_cache, in_list):
    # Use tqdm to create a progress bar
    with tqdm(total=len(in_list), desc="Processing") as pbar:
        # Iterate through each item in the list
        for _i, item in enumerate(in_list):
            prompt, safe_prompt, concept = item
            
            # Get embeddings for both prompt and concept
            if prompt not in clip_cache:
                prompt_emb = wrapClip.get_emb(prompt)
                clip_cache[prompt] = prompt_emb.cpu()
            if safe_prompt not in clip_cache:
                prompt_emb = wrapClip.get_emb(safe_prompt)
                clip_cache[safe_prompt] = prompt_emb.cpu()
            if concept not in clip_cache:
                concept_emb = wrapClip.get_emb(concept)
                clip_cache[concept] = concept_emb.cpu()
            
            # Update progress bar
            pbar.update(1)

    return clip_cache

clip_cache = {}
clip_cache = do_cache(clip_cache, train_raw_data_list)
print("Training data cached, 2 more tasks to go...")
torch.save(clip_cache, "clip_cache.pt")

clip_cache = do_cache(clip_cache, valid_raw_data_list)
print("Validation data cached, 1 more task to go...")
torch.save(clip_cache, "clip_cache.pt")

clip_cache = do_cache(clip_cache, test_raw_data_list)
print("Test data cached, all tasks completed.")
torch.save(clip_cache, "clip_cache.pt")