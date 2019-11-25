import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import Module, ModuleList
from torch.nn import Linear, LeakyReLU, PReLU, BatchNorm1d, Bilinear, LSTM
from torch.nn import BCELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.distributions import MultivariateNormal, kl_divergence
from torch import sigmoid, tanh, softmax
from torch.optim import Adam
from util import *
from config import *
from Discriminator import CodeQ
FILE_NUM = 7835
EPOCH = 5

if __name__ == '__main__':
    expert_chunk_generator = get_expert_chunk_generator()
    codeq = CodeQ().to(DEVICE)
    e_list = []
    loss_list = []

    for e in range(EPOCH*FILE_NUM):
        expert_chunk = next(expert_chunk_generator)
        expert_states = expert_chunk['states']
        expert_actions = expert_chunk['actions'].reshape((-1,))
        expert_codes = expert_chunk['codes'].reshape((-1,))
        expert_chunk_length = len(expert_states)
        expert_indice = np.arange(expert_chunk_length)
        np.random.shuffle(expert_indice)
        expert_states = expert_states[expert_indice]
        expert_actions = expert_actions[expert_indice]
        expert_codes = expert_codes[expert_indice]
        loss_sum = 0
        for i in range(expert_chunk_length//BATCH_SIZE):
            batch_expert_states = torch.as_tensor(expert_states[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.long, device=DEVICE)
            batch_expert_actions = torch.as_tensor(expert_actions[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.long, device=DEVICE)
            batch_expert_codes = expert_codes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            loss = codeq.calculate_loss(batch_expert_states, batch_expert_actions, batch_expert_codes)
            codeq.train_by_loss(loss)
            loss_sum += loss.detach().cpu().numpy()
        print(e, 'epoch: ',loss_sum/(expert_chunk_length//BATCH_SIZE))
        loss_list.append(loss_sum/(expert_chunk_length//BATCH_SIZE))
        average_list = moving_average(loss_list)
        if len(average_list) > 50:
            plt.plot(np.arange(len(average_list)), np.array(average_list))
            plt.savefig('codeQ_loss.jpg')
        if e % 1000 == 0:
            codeq.save('pretrain')
    codeq.save('pretrain')