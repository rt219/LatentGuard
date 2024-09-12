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
from configs import *
from utils import *

device = f'cuda:{device_index}'

model = EmbeddingMappingLayer(num_heads, head_dim, out_dim).to(device)
expname = f'experiment'

batch_size = 64; 

# prepare logger
cur_time = get_timestamp()
log_folder = f'logs/{cur_time}_{expname}/'
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

clip_cache = configs.clip_cache

train_dataset = OnlineDataset(train_raw_data_list, None, clip_cache=clip_cache, device = device)
valid_dataset = OnlineDataset(valid_raw_data_list, None, clip_cache= clip_cache, device = device)
eval_dataset  = OnlineDataset(test_raw_data_list, None, clip_cache=clip_cache, device = device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
eval_loader  = DataLoader(eval_dataset, batch_size=32, shuffle=False)


def train_classification(model):
    # Initialize model, loss, and optimizer

    model_save_prefix = 'model_ep'
    criterion = nn.CrossEntropyLoss()
    criterion_binary = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    mse_fn = nn.MSELoss()

    num_epochs = 1000 + 1  # Example value


    for epoch in range(num_epochs):
        if epoch % 50 == 0:
            logger.info(f"Epoch {epoch + 1} of {num_epochs}")
        if epoch % 500 == 0 and epoch != 0:
            torch.save(model, f'{log_folder}/{model_save_prefix}{epoch}.pt')
        model.train()
        num = 0
        for data,safe_data in train_loader:
            data = data.to(device)
            safe_data = safe_data.to(device)
            optimizer.zero_grad()
            _prompt  = data[:, :-1, :] # n x 78 x 768 #data[:,0, :, :] # _b*_p, 78, 768
            safe_prompt  = safe_data[:, :-1, :]       #data[:,0, :, :] # _b*_p, 78, 768
            _concept = data[:, -1, :]  # n x 768      #data[:,1, 0, :] # _b*_p, 768

            n, _l, _d = _prompt.shape
            _prompt_expanded = _prompt.unsqueeze(1).expand(-1, n, -1, -1)  # [n, n, _l, d]
            _concept_expanded = _concept.unsqueeze(0).expand(n, -1, -1)  # [n, n, d]
            safe_prompt_expanded = safe_prompt.unsqueeze(1).expand(-1, n, -1, -1)  # [n, n, _l, d]

            v_prime, q_prime = model(_prompt_expanded.reshape(-1, _l, _d), _concept_expanded.reshape(-1, _d)) # nxn, d
            v_prime2, q_prime2 = model(safe_prompt_expanded.reshape(-1, _l, _d), _concept_expanded.reshape(-1, _d)) # nxn, d
            v_prime = l2_normalize(v_prime)
            q_prime = l2_normalize(q_prime)
            v_prime2= l2_normalize(v_prime2)
            q_prime2= l2_normalize(q_prime2)
            logits = torch.sum(v_prime * q_prime, dim=1) * model.tempr
            logits = logits.view(n, n)
            
            logits_with_safe = torch.sum(v_prime2 * q_prime2, dim=1) * model.tempr
            logits_with_safe = logits_with_safe.view(n, n)

            logits_with_safe_max = logits_with_safe.max(1)[0].view(1, n)
            assert logits_with_safe_max.shape == (1, n)
            logits_with_safe_max = logits_with_safe_max.repeat(n, 1)
            logits_wmax = torch.cat([logits, logits_with_safe_max], dim=1)
            assert logits_wmax.shape == (n, n * 2)

            logits_with_safe = torch.cat([logits, logits_with_safe], dim=0)
            assert logits_with_safe.shape == (2*n, n)
            labels = torch.arange(n).to(device).detach()
            
            loss_i = F.cross_entropy(logits_wmax, labels)
            ALPHA = 0
            logits_with_safe.T[:, n:] = logits_with_safe.T[:, n:] + ALPHA
            loss_t = F.cross_entropy(logits_with_safe.T, labels)

            loss = loss_i + loss_t

            loss.backward() 
            optimizer.step()

            num += 1

        if epoch % 50 == 0: 
            logger.info(f'loss {loss.item()}: loss_i {loss_i.item()} loss_t {loss_t.item()}')
    return model


if __name__ == "__main__":
    train_classification(model)
