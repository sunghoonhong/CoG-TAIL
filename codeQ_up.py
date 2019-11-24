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
from Discriminator import CodeQ

expert_chunk_generator = get_expert_chunk_generator()
codeq = CodeQ()

for e in range(10):
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
        print(loss)

codeq.save('pretrain')