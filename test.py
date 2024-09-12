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

# prepare logger
cur_time = get_timestamp()
log_folder = f'logs/{cur_time}/'
os.system(f"mkdir -p {log_folder}")
logger = logging.getLogger(__name__)
log_file = log_folder + f"log.txt"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info(f"Training Concepts: {len(train_concepts)}")
logger.info(f"Testing Concepts: {len(test_concepts)}")
logger.info(f"Training Data List: {len(train_raw_data_list)}")
logger.info(f"Valid Data List: {len(valid_raw_data_list)}")
logger.info(f"Testing Data List: {len(test_raw_data_list)}")

device = 'cuda:0'
num_heads=16; head_dim=32; out_dim=128
model = EmbeddingMappingLayer(num_heads, head_dim, out_dim).to(device)
model.load_state_dict(torch.load("model_parameters.pth"))

batch_size = 64

# init
clip_cache = torch.load(configs.clip_cache_path)
train_dataset = OnlineDataset(train_raw_data_list, None, clip_cache=clip_cache, device = device)#wrapClip.get_emb)
valid_dataset = OnlineDataset(valid_raw_data_list, None, clip_cache= clip_cache, device = device)#wrapClip.get_emb)
eval_dataset  = OnlineDataset(test_raw_data_list, None, clip_cache=clip_cache, device = device)#wrapClip.get_emb)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #512
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
eval_loader  = DataLoader(eval_dataset, batch_size=32, shuffle=False)
wrapClip = WrapClip(device=device, model_name = 'openai/clip-vit-large-patch14')

is_train_concepts=True
eval(model, is_train_concepts, logger)
is_train_concepts=False
eval(model, is_train_concepts, logger)