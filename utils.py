import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
import random
import torch.nn.functional as F
import os
import shutil
from transformers import AutoProcessor, CLIPModel
from transformers import CLIPTokenizer
import configs
import time
import json
from sklearn.metrics import roc_auc_score

clip_cache = configs.clip_cache

class WrapClip:
    def __init__(self, device, model_name = 'openai/clip-vit-base-patch32'):
        self.device = device
        self.model_name = model_name
        self.clip_model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.model_name)

    def get_res(self, prompt):
        batch_encoding = self.clip_tokenizer(prompt, truncation=True, max_length=77, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.clip_model.device)
        outputs = self.clip_model.text_model(input_ids=tokens)

        z = outputs.last_hidden_state  
        in_token_emb = (-1,-1)
        return z, in_token_emb, batch_encoding.attention_mask, tokens

    def get_emb(self, targetp):
        with torch.no_grad():
            g_emb_transformer_tmp, (g_emb_token_tmp, _), _, target_prompt_ids_tmp = self.get_res([targetp])
            g_emb_transformer_last = g_emb_transformer_tmp[:, int(target_prompt_ids_tmp.argmax(dim=-1)), :].unsqueeze(1)
            res = torch.cat([g_emb_transformer_last, g_emb_transformer_tmp], 1)
            assert res.shape == (1, 78, 768)
            return res

def read_unsafe_file(file_path):
    with open(file_path, 'r') as f:
        lines = [it.strip() for it in f.readlines() if it.strip()!= '']
    return lines

def get_timestamp():
    timestamp = time.time()
    formatted_timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime(timestamp))
    print("formatted current stamp:", formatted_timestamp)
    return formatted_timestamp

def forward_contra_model(model, model_output):
    v_prime, q_prime= model_output
    v_prime = l2_normalize(v_prime)
    q_prime = l2_normalize(q_prime)
    dot_product = torch.sum(v_prime * q_prime, dim=1) * model.tempr
    return dot_product

# json read dataset
def prepare_dataset():
    import json
    dataset_json = json.load(open(configs.dataset_file, 'r'))
    train_concepts = dataset_json['ID_concepts']
    test_concepts = dataset_json['OOD_concepts']
    train_raw_data_list = dataset_json['ID_train_data']
    train_raw_data_list = [
        [entry['unsafe_prompt'], entry['safe_prompt'], entry['concept']]
        for entry in train_raw_data_list
    ]
    valid_raw_data_list = dataset_json['ID_test_data']
    valid_raw_data_list = [
        [entry['unsafe_prompt'], entry['safe_prompt'], entry['concept']]
        for entry in valid_raw_data_list
    ]
    test_raw_data_list = dataset_json['OOD_test_data']
    test_raw_data_list = [
        [entry['unsafe_prompt'], entry['safe_prompt'], entry['concept']]
        for entry in test_raw_data_list
    ]
    return dataset_json, train_concepts, test_concepts, train_raw_data_list, valid_raw_data_list, test_raw_data_list

dataset_json, train_concepts, test_concepts, train_raw_data_list, valid_raw_data_list, test_raw_data_list = prepare_dataset()

# Output the results
# print("Training Concepts:", len(train_concepts))
# print("Testing Concepts:", len(test_concepts))
# print("Training Data List:", len(train_raw_data_list))
# print("Valid Data List:", len(valid_raw_data_list))
# print("Testing Data List:", len(test_raw_data_list))

# prepare synonyms dict
synonyms_dict = dataset_json['concept_synonym']

# prepare adv dict 
dict_concept_adv = dataset_json['concept_adv']

device = 'cuda:0'
wrapClip = WrapClip(device=device, model_name = 'openai/clip-vit-large-patch14')

def eval_by_dict_info(
    is_train_concepts = False,
    is_unsafe = True, 
    num = 100,
    replace_dict=None, logger=None, model=None):
    global has_run, cached

    model.eval()
    pred_list = []

    from collections import defaultdict

    device = next(model.parameters()).device

    target_concept_set = train_concepts if is_train_concepts else test_concepts
    target_concept_set = list(target_concept_set)
    concept_embs = [clip_cache[concept].to(device) for concept in target_concept_set]
    all_concept_emb = torch.cat(concept_embs, dim=0).to(device)
    all_concept_emb = all_concept_emb[:, 0, :]
    
    candidate_raw_data_list = valid_raw_data_list if is_train_concepts else test_raw_data_list
    if replace_dict is not None:
        def select_items(test_list, replace_dict):
            selected_items = []
            for item in test_list:
                prompt, _, concept = item
                if concept in dict_concept_adv and concept in synonyms_dict and (concept in prompt or concept.capitalize() in prompt):
                    selected_items.append(item)
            return selected_items
        selected_items = select_items(candidate_raw_data_list, replace_dict)
    else:
        selected_items = candidate_raw_data_list

    logger.info(f'Number of to be evaluated prompts: {len(selected_items)}')
    for _i, prompt_data in enumerate(selected_items):
        if _i%2000==0:
            logger.info(f'process {_i}')
        cur_concept = prompt_data[2]
        if replace_dict is not None:
            advp = replace_dict[cur_concept]
            if type(advp) is list:
                advp = advp[0]
        prompt = prompt_data[0] if is_unsafe else prompt_data[1]
        if is_unsafe and replace_dict is not None:
            if cur_concept in prompt:
                prompt = prompt.replace(cur_concept, advp)
            elif cur_concept.capitalize() in prompt:
                prompt = prompt.replace(cur_concept.capitalize(), advp)
            else:
                logger.info(f'warn {prompt} {cur_concept}')
        if prompt in clip_cache:
            prompt_emb = clip_cache[prompt].to(device)
        else:
            prompt_emb = wrapClip.get_emb(prompt).to(device)
            clip_cache[prompt]=prompt_emb
        with torch.no_grad():
            prompt_emb = prompt_emb.to(device)
            repeated_prompt_emb = prompt_emb.repeat(len(target_concept_set), 1, 1)
            output = model(repeated_prompt_emb.to(device), all_concept_emb.to(device))
            dot_product = forward_contra_model(model, output)
            
            predicted_maxv = dot_product.max(0)[0].cpu().numpy()
            pred_list.append(predicted_maxv)

    return selected_items, pred_list


def eval(model, is_train_concepts=True, logger=None):
    model.eval()
    def l2_normalize(tensor, axis=1):
        return F.normalize(tensor, p=2, dim=axis)
    eval_func = eval_by_dict_info
    with torch.no_grad():
        def cal_auc(replace_dict, num = 1000):
            _, pred_harm  = eval_func(is_train_concepts = is_train_concepts,  is_unsafe= True,   num = num, replace_dict=replace_dict, logger=logger, model=model)
            _, pred_safe  = eval_func(is_train_concepts = is_train_concepts,  is_unsafe= False,  num = num, replace_dict=replace_dict, logger=logger, model=model)
            pred = pred_harm + pred_safe
            gt = [1 for it in range(len(pred_harm))] + [0 for it in range(len(pred_safe))]
            # calculate the AUC
            res = roc_auc_score(gt, pred)
            logger.info(f'AUC {res}')

        logger.info(f'[eval] Eval on explicit cases')
        cal_auc(None)
        logger.info(f'[eval] Eval on synonyms cases')
        cal_auc(synonyms_dict)
        logger.info(f'[eval] Eval on adversarial cases')
        cal_auc(dict_concept_adv)

class OnlineDataset(Dataset):
    def __init__(self, data_list, emb_func, clip_cache, device):
        self.data_list = data_list
        self.emb_func = emb_func
        self.neg_num = 1000
        self.data_dict = {}
        for prompt, safe_prompt, concept in data_list:
            if concept in self.data_dict:
                self.data_dict[concept].append([prompt, safe_prompt])
            else:
                self.data_dict[concept] = [[prompt, safe_prompt]]
        self.concepts = list(self.data_dict.keys())
        self.clip_cache = clip_cache
        self.device = device

    def __len__(self):
        return len(self.concepts)

    def __getitem__(self, idx):
        # Get the item and embeddings
        concept = self.concepts[idx]
        item = random.sample(self.data_dict[concept], 1)[0]
        prompt, safe_prompt = item
        eb1 = self.get_embedding(prompt)[0]
        eb_safe = self.get_embedding(safe_prompt)[0]
        eb2 = self.get_embedding(concept)[0, 0:1]
        assert eb1.shape == (78, 768)
        assert eb2.shape == (1, 768)

        # Concatenate embeddings to form positive data
        pos_data = torch.cat((eb1, eb2), dim=0)
        safe_data = torch.cat((eb_safe, eb2), dim=0)
        assert pos_data.shape == (79, 768)
        assert safe_data.shape == (79, 768)
        return pos_data, safe_data
        

    def get_embedding(self, text):
        # Generate embedding for the given text
        if text in self.clip_cache:
            return self.clip_cache[text].to(self.device).detach()
        print("[warn] out of cache!")
        emb = self.emb_func(text)
        return emb.to(self.device).detach()

    def get_negative_sample(self, current_idx, concept, c_emb):
        # Randomly select another item
        idx = random.choice(range(len(self.data_list)))
        target_num = 100 #1000//32
        neg_data_list = []
        neg_label_list = []
        while len(neg_data_list) < target_num:
            random_elements = random.sample(self.data_list, target_num)
            for _p, _c in random_elements:
                if _c != concept:
                    item = [_p, _c]
                    eb1 = self.get_embedding(_p)
                    eb2 = c_emb
                    neg_data = torch.cat((eb1, eb2), dim=0)
                    neg_label = torch.tensor(-1)
                    neg_data_list.append(neg_data.unsqueeze(0))
                    neg_label_list.append(neg_label.unsqueeze(0))
                    if len(neg_data_list) >= target_num:
                        break
        neg_data_list = torch.cat(neg_data_list, dim=0)
        neg_label_list = torch.cat(neg_label_list, dim=0)
        return neg_data_list, neg_label_list


class EmbeddingMappingLayer(nn.Module):
    def __init__(self, num_heads, head_dim, out_dim=512):
        super(EmbeddingMappingLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.key_d = self.head_dim * self.num_heads
        self.out_dim = out_dim

        # Ensure the head dimension is an integer
        assert self.key_d % self.num_heads == 0, "key_d must be divisible by num_heads"

        self.x1_to_key = nn.Linear(768, self.key_d)
        self.x2_to_query = nn.Linear(768, self.key_d)
        self.x1_to_value = nn.Linear(768, self.key_d)
        
        self.final_mlp = nn.Linear(self.key_d, self.out_dim)  # Optional: final layer to combine head outputs
        self.mlp_query1 = nn.Linear(self.key_d, self.out_dim)
        self.tempr = nn.Parameter(torch.tensor(1/0.07), requires_grad=True) #1.0 / 0.07 #

    def forward(self, x1, x2):
        batch_size, seq_len, _ = x1.shape

        # Process x1 to generate keys and values
        # key shape: (batch_size, seq_len, num_heads, head_dim)
        key = self.x1_to_key(x1).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.transpose(1, 2)  # Reshape to (batch_size, num_heads, seq_len, head_dim)

        # value shape: (batch_size, seq_len, num_heads, head_dim)
        value = self.x1_to_value(x1).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.transpose(1, 2)  # Reshape to (batch_size, num_heads, seq_len, head_dim)

        # Process x2 to generate queries
        # query shape: (batch_size, 1, num_heads, head_dim)
        query = self.x2_to_query(x2).view(batch_size, 1, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)  # Reshape to (batch_size, num_heads, 1, head_dim)

        # Compute attention scores for each head
        # attention_scores shape: (batch_size, num_heads, 1, seq_len)
        scaling_factor = self.head_dim ** 0.5
        attention_scores = torch.einsum('bnqd,bnkd->bnqk', query, key) / scaling_factor
        attention_weights = F.softmax(attention_scores, dim=-1)

        V = torch.einsum('bnqk,bnkd->bnqd', attention_weights, value)
        V = V.view(batch_size, -1)
        V = self.final_mlp(V)
        V_prime = V  # V' after processing

        query = query.view(batch_size, -1)

        query = self.mlp_query1(query)
        query_prime = query  # query' after processing

        return V_prime, query_prime

def l2_normalize(tensor, axis=1):
    return F.normalize(tensor, p=2, dim=axis)


