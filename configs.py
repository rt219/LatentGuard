dataset_file = './dataset/CoPro_v1.0.json'
clip_cache_path="clip_cache.pt"

import torch
import os
if os.path.exists(clip_cache_path):
    clip_cache = torch.load(clip_cache_path)
else:
    print("[warning] clip_cache_path does not exist. please create a new cache file if you want to train or test on CoPro. you can ignore this warning if you just run inference.py.")
    clip_cache = None

device_index = '0'
num_heads = 16
head_dim = 32
out_dim = 128