import numpy as np
import torch
from torch.nn import Module, ModuleList
from torch.nn import Linear, LeakyReLU, PReLU, BatchNorm1d, Bilinear, LSTM
from torch.nn import BCELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.distributions import MultivariateNormal, kl_divergence
from torch import sigmoid, tanh, softmax
from torch.optim import Adam
from util import *
from config import *
from Discriminator import Discriminator

FAKE_DIR = './bpe_fake_token'
FAKE_CHUNKS_NUM = 200
FAKE_CHUNK_LENGTH = 512

def get_fake_chunk_generator():
    chunk_list = get_fake_chunk_list()
    while True:
        for chunk in chunk_list:
            yield chunk
        chunk_list = get_fake_chunk_list()

def get_fake_chunk_list():
    fake_files = os.listdir(FAKE_DIR)
    file_lists = random.sample(fake_files, FAKE_CHUNKS_NUM)
    states_list = []
    actions_list = []
    codes_list = []
    for file_name in file_lists:
        file_path = os.path.join(FAKE_DIR, file_name)
        chunk = np.load(file_path)
        states_list.append(chunk['states'])
        actions_list.append(chunk['actions'])
        codes_list.append(chunk['codes'])
    states = np.concatenate(states_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    codes = np.concatenate(codes_list, axis=0)
    chunk_length = len(states)
    indice = np.arange(chunk_length)
    np.random.shuffle(indice)
    states = states[indice]
    actions = actions[indice]
    codes = codes[indice]
    chunk_list = []
    for i in range(chunk_length//FAKE_CHUNK_LENGTH):
        tmp = dict()
        tmp['states'] = states[i*FAKE_CHUNK_LENGTH:(i+1)*FAKE_CHUNK_LENGTH]
        tmp['actions'] = actions[i*FAKE_CHUNK_LENGTH:(i+1)*FAKE_CHUNK_LENGTH]
        tmp['codes'] = codes[i*FAKE_CHUNK_LENGTH:(i+1)*FAKE_CHUNK_LENGTH]
        chunk_list.append(tmp)
    return chunk_list


expert_chunk_generator = get_expert_chunk_generator()
fake_chunk_generator = get_fake_chunk_generator()
disc = Discriminator()

for e in range(10):
    expert_chunk = next(expert_chunk_generator)
    expert_states = expert_chunk['states']
    expert_actions = expert_chunk['actions'].reshape((-1,))
    expert_chunk_length = len(expert_states) // 2
    expert_indice = np.arange(expert_chunk_length)
    np.random.shuffle(expert_indice)
    expert_states = expert_states[expert_indice]
    expert_actions = expert_actions[expert_indice]

    fake_chunk = next(fake_chunk_generator)
    fake_chunk = next(fake_chunk_generator)
    fake_states = fake_chunk['states']
    fake_actions = fake_chunk['actions'].reshape((-1,))
    fake_chunk_length = len(fake_states) // 2
    fake_indice = np.arange(fake_chunk_length)
    np.random.shuffle(fake_indice)
    fake_states = fake_states[fake_indice]
    fake_actions = fake_actions[fake_indice]

    tmp_chunk_length = fake_chunk_length + expert_chunk_length
    tmp_states = np.vstack([fake_states,expert_states])
    tmp_actions = np.hstack([fake_actions, expert_actions])

    for i in range(tmp_chunk_length//BATCH_SIZE):
        batch_tmp_states = torch.as_tensor(tmp_states[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.long, device=DEVICE)
        batch_tmp_actions = torch.as_tensor(tmp_actions[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.long, device=DEVICE)
        loss = disc.calculate_vail_loss(batch_tmp_states, batch_tmp_actions, 0.01)[0]
        disc.train_by_loss(loss)
        print(loss)