import torch
import random
import os
import numpy as np
from transformers import BertTokenizer, BertModel
from config import *

def get_random_action(obs):
    return random.randint(0, VOCAB_SIZE)

def to_onehot(c):
    '''
    IN:
    c: single integer
    OR
    c: [BATCH_SIZE,](list)(ndarray)
    OUT:
    if c is a single integer:
        one_hot_c: [CODE_SIZE,](torch.FloatTensor)
    elif c is a list:
        one_hot_c: [BATCH_SIZE, CODE_SIZE](torch.FloatTensor)
    '''
    tmp = torch.eye(CODE_SIZE)
    return tmp[c].to(DEVICE)

def get_bert_model_and_tokenizer():
    model = BertModel.from_pretrained(PRETRAINED_WEIGHTS).eval()
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_WEIGHTS)
    return model.to(DEVICE), tokenizer

def tmp_get_expert_chunk():
    return np.load('./expert_data/tmp.npz')

def get_expert_chunk():
    expert_files = os.listdir(EXPERT_DIR)
    file_path = os.path.join(EXPERT_DIR, random.sample(expert_files, 1)[0])
    return np.load(file_path)


if __name__ == '__main__':
    a = get_expert_chunk()
    print(a['states'].shape)
    print(a['actions'].shape)
    print(a['codes'].shape)