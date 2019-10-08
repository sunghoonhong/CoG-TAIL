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

def get_expert_chunk_generator():
    chunk_list = get_expert_chunk_list()
    print(len(chunk_list))
    while True:
        for chunk in chunk_list:
            yield chunk
        chunk_list = get_expert_chunk_list()

def get_expert_chunk_list():
    expert_files = os.listdir(EXPERT_DIR)
    file_lists = random.sample(expert_files, EXPERT_CHUNKS_NUM)
    states_list = []
    actions_list = []
    action_ids_list = []
    codes_list = []
    for file_name in file_lists:
        file_path = os.path.join(EXPERT_DIR, file_name)
        chunk = np.load(file_path)
        states_list.append(chunk['states'])
        actions_list.append(chunk['actions'])
        action_ids_list.append(chunk['actions_ids'])
        codes_list.append(chunk['codes'])
    states = np.concatenate(states_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    action_ids = np.concatenate(action_ids_list, axis=0)
    codes = np.concatenate(codes_list, axis=0)
    chunk_length = len(states)
    indice = np.arange(chunk_length)
    np.random.shuffle(indice)
    states = states[indice]
    actions = actions[indice]
    action_ids = action_ids[indice]
    codes = codes[indice]
    chunk_list = []
    for i in range(chunk_length//EXPERT_CHUNK_LENGTH):
        tmp = dict()
        tmp['states'] = states[i*EXPERT_CHUNK_LENGTH:(i+1)*EXPERT_CHUNK_LENGTH]
        tmp['actions'] = actions[i*EXPERT_CHUNK_LENGTH:(i+1)*EXPERT_CHUNK_LENGTH]
        tmp['action_ids'] = action_ids[i*EXPERT_CHUNK_LENGTH:(i+1)*EXPERT_CHUNK_LENGTH]
        tmp['codes'] = codes[i*EXPERT_CHUNK_LENGTH:(i+1)*EXPERT_CHUNK_LENGTH]
        chunk_list.append(tmp)
    return chunk_list


if __name__ == '__main__':
    g = get_expert_chunk_generator()
    for i in range(100):
        a = next(g)
        print(i)
        print(a['states'].shape)
        print(a['actions'].shape)
        print(a['codes'].shape)