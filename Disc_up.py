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

import sys
FAKE_DIR = './bpe_fake_token'
FAKE_CHUNKS_NUM = 50
FAKE_CHUNK_LENGTH = 512

def get_agent_chunk_generator():
    chunk_list = get_agent_chunk_list()
    while True:
        for chunk in chunk_list:
            yield chunk
        chunk_list = get_agent_chunk_list()

def get_agent_chunk_list():
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

if __name__ == '__main__':
    expert_chunk_generator = get_expert_chunk_generator()
    agent_chunk_generator = get_agent_chunk_generator()
    disc = Discriminator().to(DEVICE)
    half_batch_size = int(BATCH_SIZE/2)
    for e in range(400):
        print('e: ', e)
        expert_chunk = next(expert_chunk_generator)
        expert_states = expert_chunk['states']
        expert_actions = expert_chunk['actions'].reshape((-1,))
        expert_chunk_length = len(expert_states)
        expert_indice = np.arange(expert_chunk_length)
        np.random.shuffle(expert_indice)
        expert_states = expert_states[expert_indice]
        expert_actions = expert_actions[expert_indice]

        agent_chunk = next(agent_chunk_generator)
        agent_states = agent_chunk['states']
        agent_actions = agent_chunk['actions'].reshape((-1,))
        agent_chunk_length = len(agent_states)
        agent_indice = np.arange(agent_chunk_length)
        np.random.shuffle(agent_indice)
        agent_states = agent_states[agent_indice]
        agent_actions = agent_actions[agent_indice]

        for i in range(min(expert_chunk_length//half_batch_size, agent_chunk_length//half_batch_size)):
            #agent
            batch_agent_states = torch.as_tensor(agent_states[i*half_batch_size:(i+1)*half_batch_size], dtype=torch.float, device=DEVICE)
            batch_agent_actions = torch.as_tensor(agent_actions[i*half_batch_size:(i+1)*half_batch_size], dtype=torch.float, device=DEVICE)
            #expert
            batch_expert_states = torch.as_tensor(expert_states[i*half_batch_size:(i+1)*half_batch_size], dtype=torch.float, device=DEVICE)
            batch_expert_actions = torch.as_tensor(expert_actions[i*half_batch_size:(i+1)*half_batch_size], dtype=torch.float, device=DEVICE)
            #to make same len
            min_length = min(len(batch_agent_states), len(batch_expert_states))
            batch_agent_states = batch_agent_states[:min_length]
            batch_agent_actions = batch_agent_actions[:min_length]
            batch_expert_states = batch_expert_states[:min_length]
            batch_expert_actions = batch_expert_actions[:min_length]
            assert len(batch_agent_states) == len(batch_expert_states)
            #concat
            batch_states = torch.cat((batch_agent_states, batch_expert_states), 0)
            batch_actions = torch.cat((batch_agent_actions, batch_expert_actions), 0)
            loss = disc.calculate_wail_loss(batch_states, batch_actions)
            disc.train_by_loss(loss)
            kl_coef = max(0, kl_coef + KL_STEP*(kl - IC))
            print(loss, kl_coef)
        disc.save('pretrain')