import torch
from time import time
import numpy as np
import random
from torch.nn import Module, ModuleList
from torch.nn import Linear, PReLU, BatchNorm1d, Sigmoid
from torch.distributions import Categorical
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam
from config import *
from util import *

class Actor_Critic(Module):
    def __init__(self, pretrain_loss_function=None):
        super().__init__()
        last_hidden_layer_input = AC_HIDDEN_UNIT_NUM + AC_HIDDEN_UNIT_STRIDE*(AC_HIDDEN_LAYER_NUM - 1)
        self.l1 = Linear(STATE_SIZE*CODE_SIZE, AC_HIDDEN_UNIT_NUM)
        self.b1 = BatchNorm1d(AC_HIDDEN_UNIT_NUM)
        self.a1 = PReLU()
        self.hidden_layers = ModuleList(
            [Linear(i, i + AC_HIDDEN_UNIT_STRIDE) for i in range(AC_HIDDEN_UNIT_NUM, last_hidden_layer_input + 1, AC_HIDDEN_UNIT_STRIDE)]
        )
        self.batchnorm_layers = ModuleList(
            [BatchNorm1d(i + AC_HIDDEN_UNIT_STRIDE) for i in range(AC_HIDDEN_UNIT_NUM, last_hidden_layer_input + 1, AC_HIDDEN_UNIT_STRIDE)]
        )
        self.activation_layers = ModuleList([PReLU() for _ in range(AC_HIDDEN_LAYER_NUM)])
        self.actor_layer = Linear(last_hidden_layer_input + AC_HIDDEN_UNIT_STRIDE, VOCAB_SIZE)
        self.critic_hidden_layer = Linear(last_hidden_layer_input + AC_HIDDEN_UNIT_STRIDE, CRITIC_HIDDEN_UNIT_NUM)
        self.critic_batchnorm = BatchNorm1d(CRITIC_HIDDEN_UNIT_NUM)
        self.critic_activation = PReLU()
        self.critic_layer = Linear(CRITIC_HIDDEN_UNIT_NUM, 1)
        self.critic_loss_function = MSELoss()
        self.opt = Adam(self.parameters(), lr=AC_LR)
        if pretrain_loss_function is not None:
            self.pretrain_loss_function = pretrain_loss_function
            self.pretrain_opt = Adam(self.parameters(), lr=PRETRAIN_LR)

    def forward(self, s, c):
        '''
        IN:
        s: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor)
        c: [BATCH_SIZE, CODE_SIZE](torch.FloatTensor)
        OUT:
        action_score = [BATCH_SIZE, VOCAB_SIZE]
        '''
        s_c = torch.bmm(s.unsqueeze(2), c.unsqueeze(1)).view(-1, STATE_SIZE*CODE_SIZE)
        x = self.a1(self.l1(s_c))
        for layer, batchnorm, activation in zip(self.hidden_layers, self.batchnorm_layers, self.activation_layers):
            x = activation(batchnorm((layer(x))))
        x = self.actor_layer(x)
        return x

    def action_forward(self, s, c, a=None):
        '''
        IN:
        s: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor)
        c: [BATCH_SIZE, CODE_SIZE](torch.FloatTensor)
        a: [BATCH_SIZE, ] (index number)(torch.LongTensor)
        OUT:
        IF a is not provided:
            sampled_actions: [BATCH_SIZE,](torch.LongTensor)
            sampled_actions_log_probs: [BATCH_SIZE,](torch.FloatTensor)(detached)
        ELSE:
            actions_log_probs: [BATCH_SIZE,](torch.FloatTensor)
            entropy: [BATCH_SIZE,](torch.FloatTensor)
        '''
        s_c = torch.bmm(s.unsqueeze(2), c.unsqueeze(1)).view(-1, STATE_SIZE*CODE_SIZE)
        x = self.a1(self.l1(s_c))
        for layer, batchnorm, activation in zip(self.hidden_layers, self.batchnorm_layers, self.activation_layers):
            x = activation(batchnorm((layer(x))))
        x = self.actor_layer(x)
        
        top_values, _ = x.topk(TOP_K)
        min_values = top_values[:,-1].unsqueeze(1)
        x = torch.where(x < min_values, torch.full(x.size(), fill_value=-15.0, device=DEVICE), x)
        
        dist = Categorical(logits=x)
        if a is None:
            sampled = dist.sample()
            log_prob = dist.log_prob(sampled)
            return sampled, log_prob.detach()
        else:
            return dist.log_prob(a), dist.entropy() #initial entropy := 3.6xxxx

    def critic_forward(self, s, c):
        '''
        IN:
        s: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor)
        c: [BATCH_SIZE, CODE_SIZE](torch.FloatTensor)
        OUT:
        value: [BATCH_SIZE, 1]
        '''
        s_c = torch.bmm(s.unsqueeze(2), c.unsqueeze(1)).view(-1, STATE_SIZE*CODE_SIZE)
        x = self.a1(self.l1(s_c))
        for layer, batchnorm, activation in zip(self.hidden_layers, self.batchnorm_layers, self.activation_layers):
            x = activation(batchnorm((layer(x))))
        x = self.critic_activation(self.critic_batchnorm(self.critic_hidden_layer(x)))
        x = self.critic_layer(x)
        return x

    def critic_loss(self, states, codes, oracle_values):
        '''
        IN:
        states: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor)
        codes: [BATCH_SIZE,](list)(ndarray)
        OUT:
        Loss
        '''
        codes = to_onehot(codes)
        out = self.critic_forward(states, codes).view(-1)
        return self.critic_loss_function(out, oracle_values)

    def actor_loss(self, states, actions, codes, gaes, old_log_probs):
        '''
        states: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor)
        actions: [BATCH_SIZE,](torch.LongTensor)
        codes: [BATCH_SIZE,](list)(ndarray)
        gaes: [BATCH_SIZE,](torch.FloatTensor)
        old_log_probs: [BATCH_SIZE,](torch.FloatTensor)
        '''
        codes = to_onehot(codes)
        now_log_probs, entropy = self.action_forward(states, codes, actions)
        r = torch.exp(now_log_probs - old_log_probs)
        tmp = torch.min(r*gaes, torch.clamp(r, 1-EPSILON, 1+EPSILON)*gaes)
        minus_entropy = -torch.mean(entropy)
        print('minus_entropy: ', minus_entropy.cpu())
        loss = -torch.mean(tmp) + ENTROPY*minus_entropy
        return loss

    def pretrain_loss(self, states, action_ids, codes):
        '''
        IN:
        states: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor)
        action_ids: [BATCH_SIZE,](torch.LongTensor)
        codes: [BATCH_SIZE,](list)(ndarray)
        pretrain_loss_function: CrossEntropyLoss with weights
        '''
        codes = to_onehot(codes)
        action_score = self.forward(states, codes)
        loss = self.pretrain_loss_function(action_score, action_ids)
        return loss

    def train_by_loss(self, loss):
        self.zero_grad()
        loss.backward()
        self.opt.step()

    def pretrain_by_loss(self, loss):
        self.zero_grad()
        loss.backward()
        self.pretrain_opt.step()

if __name__ == '__main__':
    tmp_batch_size = 4
    tmp_state = torch.randn(tmp_batch_size, STATE_SIZE)
    tmp_action = [random.randint(0, VOCAB_SIZE-1) for _ in range(tmp_batch_size)]
    tmp_code = [random.randint(0,1) for _ in range(tmp_batch_size)]
    tmp_gae = [random.random() for _ in range(tmp_batch_size)]
    tmp_prob = [random.random() for _ in range(tmp_batch_size)]
    ac = Actor_Critic()
    print(ac.actor_loss(tmp_state, tmp_action, tmp_code, tmp_gae,tmp_prob))

