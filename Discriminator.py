import numpy as np
import torch
from torch.nn import Module, ModuleList
from torch.nn import Linear, LeakyReLU, PReLU, BatchNorm1d
from torch.nn import BCELoss, CrossEntropyLoss
from torch import sigmoid
from torch.optim import Adam
from config import *

"""
input: (s, a) pair
output: score of reality, code reconstruction
"""

class Discriminator(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(STATE_SIZE + COMPRESSED_VOCAB_SIZE, DISC_HIDDEN_UNIT_NUM)
        self.a1 = PReLU()
        self.hidden_layers = ModuleList([Linear(DISC_HIDDEN_UNIT_NUM, DISC_HIDDEN_UNIT_NUM) for _ in range(DISC_HIDDEN_LAYER_NUM)])
        self.batchnorm_layers = ModuleList([BatchNorm1d(DISC_HIDDEN_UNIT_NUM) for _ in range(DISC_HIDDEN_LAYER_NUM)])
        self.activation_layers = ModuleList([PReLU() for _ in range(DISC_HIDDEN_LAYER_NUM)])
        self.disc_out = Linear(DISC_HIDDEN_UNIT_NUM, 1)
        self.code_out = Linear(DISC_HIDDEN_UNIT_NUM, CODE_SIZE)
        self.disc_loss = BCELoss()
        self.code_loss = CrossEntropyLoss()
        self.opt = Adam(self.parameters(), lr=DISC_LR)


    def forward(self, s, a):
        '''
        IN:
        s: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor)
        a: [BATCH_SIZE, COMPRESSED_VOCAB_SIZE](torch.FloatTensor)
        OUT:
        (disc_out, code_out) where
            disc_out: [BATCH_SIZE, 1](torch.FloatTensor)(sigmoid_normalized)
            code_out: [BATCH_SIZE, CODE_SIZE](torch.FloatTensor)(unnormalized_score)
        '''
        s_a = torch.cat((s, a), dim=1)
        x = self.a1(self.l1(s_a))
        for layer, batchnorm, activation in zip(self.hidden_layers, self.batchnorm_layers, self.activation_layers):
            x = activation(batchnorm((layer(x))))
        disc_out = sigmoid(self.disc_out(x))
        code_out = self.code_out(x)
        return disc_out, code_out

    def calculate_loss(self, s, a, is_agent, code_answer, verbose=False):
        '''
        IN:
        s: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor)
        a: [BATCH_SIZE, COMPRESSED_VOCAB_SIZE](torch.FloatTensor)
        is_agent: [BATCH_SIZE,](torch.FloatTensor)
        code_answer: [BATCH_SIZE,](ndarray)
        '''
        self.train()
        is_agent = is_agent.view(-1, 1)
        code_answer = torch.as_tensor(code_answer, dtype=torch.long, device=DEVICE)
        disc_out, code_out = self.forward(s, a)
        if verbose:
            print('disc_out:', disc_out.view(-1,))
        disc_loss = self.disc_loss(disc_out, is_agent)
        code_loss = self.code_loss(code_out, code_answer)
        return disc_loss + WEIGHT_FOR_CODE*code_loss

    def train_by_loss(self, loss):
        loss.backward()
        self.opt.step()

if __name__ == '__main__':
    tmp_batch = 10
    tmp_state = torch.randn(tmp_batch, STATE_SIZE)
    tmp_vocab = torch.randn(tmp_batch, COMPRESSED_VOCAB_SIZE)
    is_agent = torch.ones(tmp_batch, dtype=torch.float)
    code_answer = [0]*tmp_batch
    disc = Discriminator()
    loss = disc.calculate_loss(tmp_state, tmp_vocab, is_agent, code_answer)
    print(loss)
    disc.train_by_loss(loss)