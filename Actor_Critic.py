import torch
from time import time
import numpy as np
import random
from torch.nn import Module, ModuleList
from torch.nn import Linear, PReLU, BatchNorm1d, Sigmoid, LSTM
from torch.distributions import Categorical
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam
from config import *
from util import *

class Actor_Critic(Module):
    def __init__(self, pretrain_loss_function=None):
        super().__init__()
        self.embed = Linear(VOCAB_SIZE, EMB_SIZE, bias=False)
        self.lstm = LSTM(EMB_SIZE, AC_HIDDEN_UNIT_NUM, num_layers=AC_HIDDEN_LAYER_NUM,
                bias=True, batch_first=True, bidirectional=True)
        self.expand_c = Linear(CODE_SIZE, AC_LAST2_UNIT_NUM)
        self.l1 = Linear(AC_HIDDEN_UNIT_NUM*2, AC_HIDDEN_UNIT_NUM)
        self.a1 = PReLU()
        self.l2 = Linear(AC_HIDDEN_UNIT_NUM, AC_LAST2_UNIT_NUM)
        self.a2 = PReLU()
        # self.hidden_layers = ModuleList(
        #     [Linear(AC_HIDDEN_UNIT_NUM, AC_LAST2_UNIT_NUM) for i in range(AC_HIDDEN_LAYER_NUM)]
        # )
        # self.activation_layers = ModuleList([PReLU() for _ in range(AC_HIDDEN_LAYER_NUM)])
        self.actor_layer = Linear(AC_LAST2_UNIT_NUM, VOCAB_SIZE)

        self.critic_hidden_layer = Linear(AC_LAST2_UNIT_NUM, CRITIC_HIDDEN_UNIT_NUM)
        self.critic_activation = PReLU()
        self.critic_layer = Linear(CRITIC_HIDDEN_UNIT_NUM, 1)
        self.critic_loss_function = MSELoss()
        self.opt = Adam(self.parameters(), lr=AC_LR)
        if pretrain_loss_function is not None:
            self.pretrain_loss_function = pretrain_loss_function
            self.pretrain_opt = Adam(self.parameters(), lr=PRETRAIN_LR)

    def forward(self, s, c, test=False):
        '''
        IN:
        s: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor)
        c: [BATCH_SIZE, CODE_SIZE](torch.FloatTensor)
        OUT:
        action_logits = [BATCH_SIZE, VOCAB_SIZE]
        '''
        
        # s_c = torch.bmm(s.unsqueeze(2), c.unsqueeze(1)).view(-1, STATE_SIZE*CODE_SIZE)
        # x = self.a1(self.l1(s_c))
        embed = self.embed(to_onehot_vocab(s).view(-1, GEN_MAX_LEN, VOCAB_SIZE))
        lstm_out = self.lstm(embed)[0][:, -1]
        reduced_s = self.a1(self.l1(lstm_out))
        reduced_s = self.a2(self.l2(reduced_s))
        expanded_c = self.expand_c(c)
        x = reduced_s + expanded_c
        x = self.actor_layer(x)
        
        if test:
            top_values, _ = x.topk(TOP_K)
            min_values = top_values[:,-1].unsqueeze(1)
            x = torch.where(x < min_values, torch.full(x.size(), fill_value=-15.0, device=DEVICE), x)
        
        return x

    def action_forward(self, s, c, a=None, test=False):
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
        # s_c = torch.bmm(s.unsqueeze(2), c.unsqueeze(1)).view(-1, STATE_SIZE*CODE_SIZE)
        s = to_onehot_vocab(s).view(-1, GEN_MAX_LEN, VOCAB_SIZE)
        embed = self.embed(s)
        lstm_out = self.lstm(embed)[0][:, -1]
        reduced_s = self.a1(self.l1(lstm_out))
        reduced_s = self.a2(self.l2(reduced_s))
        expanded_c = self.expand_c(c)
        x = reduced_s + expanded_c
        x = self.actor_layer(x)

        if test:
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
        # s_c = torch.bmm(s.unsqueeze(2), c.unsqueeze(1)).view(-1, STATE_SIZE*CODE_SIZE)
        # x = self.a1(self.l1(s_c))
        embed = self.embed(to_onehot_vocab(s).view(-1, GEN_MAX_LEN, VOCAB_SIZE))
        lstm_out = self.lstm(embed)[0][:, -1]
        reduced_s = self.a1(self.l1(lstm_out))
        reduced_s = self.a2(self.l2(reduced_s))
        expanded_c = self.expand_c(c)
        x = reduced_s + expanded_c
        x = self.critic_activation(self.critic_hidden_layer(x))
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

    def pretrain_loss(self, states, actions, codes):
        '''
        IN:
        states: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor)
        actions: [BATCH_SIZE,](torch.LongTensor)
        codes: [BATCH_SIZE,](list)(ndarray)
        pretrain_loss_function: CrossEntropyLoss with weights
        '''
        codes = to_onehot(codes)
        action_score = self.forward(states, codes)
        loss = self.pretrain_loss_function(action_score, actions)
        return loss

    def train_by_loss(self, loss):
        self.zero_grad()
        loss.backward()
        self.opt.step()

    def pretrain_by_loss(self, loss):
        self.zero_grad()
        loss.backward()
        self.pretrain_opt.step()

    def save(self, epoch):
        path = MODEL_SAVEPATH + str(epoch) + "ac" + ".pt"
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    tmp_batch_size = 4
    tmp_state = torch.randn(tmp_batch_size, STATE_SIZE).long()
    tmp_action = torch.randint(VOCAB_SIZE, (tmp_batch_size,))
    tmp_code = [random.randint(0,1) for _ in range(tmp_batch_size)]
    tmp_gae = torch.rand((tmp_batch_size,))
    tmp_prob = torch.rand((tmp_batch_size,))
    ac = Actor_Critic()
    print(ac.actor_loss(tmp_state, tmp_action, tmp_code, tmp_gae,tmp_prob))

