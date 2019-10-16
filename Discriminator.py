import numpy as np
import torch
from torch.nn import Module, ModuleList
from torch.nn import Linear, LeakyReLU, PReLU, BatchNorm1d
from torch.nn import BCELoss, CrossEntropyLoss
from torch.distributions import MultivariateNormal, kl_divergence
from torch import sigmoid, tanh
from torch.optim import Adam
from config import *

"""
input: (s, a) pair
output: score of reality, code reconstruction
"""

class Discriminator(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(STATE_SIZE*COMPRESSED_VOCAB_SIZE, DISC_HIDDEN_UNIT_NUM)
        self.a1 = PReLU()
        self.l2 = Linear(DISC_HIDDEN_UNIT_NUM, AUTOENCODER_HIDDEN_UNIT_NUM)
        self.a2 = PReLU()
        self.l3 = Linear(AUTOENCODER_HIDDEN_UNIT_NUM, 2*DISC_LATENT_SIZE)
        self.disc_out = Linear(DISC_LATENT_SIZE, 1)
        self.code_out = Linear(DISC_LATENT_SIZE, CODE_SIZE)
        self.disc_loss = BCELoss()
        self.code_loss = CrossEntropyLoss()
        target_mean = torch.zeros(1, COMPRESSED_VOCAB_SIZE).to(DEVICE)
        target_cov = torch.diag_embed(torch.ones(1, COMPRESSED_VOCAB_SIZE)).to(DEVICE)
        self.target_dist = MultivariateNormal(target_mean, target_cov)
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
        s_a = torch.bmm(s.unsqueeze(2), a.unsqueeze(1)).view(-1, STATE_SIZE*COMPRESSED_VOCAB_SIZE)
        x = self.a1(self.l1(s_a))
        x = self.l3(self.a2(self.l2(x)))
        m = x[:, :COMPRESSED_VOCAB_SIZE]
        cov = torch.diag_embed(torch.exp(x[:, COMPRESSED_VOCAB_SIZE:]))
        dist = MultivariateNormal(m, cov)
        latent_variable = dist.rsample()
        disc_out = sigmoid(self.disc_out(latent_variable))
        code_out = self.code_out(latent_variable)
        return disc_out, code_out

    def get_distribution(self, s, a):
        '''
        IN:
        s: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor)
        a: [BATCH_SIZE, COMPRESSED_VOCAB_SIZE](torch.FloatTensor)
        OUT:
        dist: torch.distributions.MultivariateNormal where
            m=[BATCH_SIZE, LATENT_SIZE]
        '''
        s_a = torch.bmm(s.unsqueeze(2), a.unsqueeze(1)).view(-1, STATE_SIZE*COMPRESSED_VOCAB_SIZE)
        x = self.a1(self.l1(s_a))
        x = self.l3(self.a2(self.l2(x)))
        m = x[:, :COMPRESSED_VOCAB_SIZE]
        cov = torch.diag_embed(torch.exp(x[:, COMPRESSED_VOCAB_SIZE:]))
        dist = MultivariateNormal(m, cov)
        return dist

    def calculate_loss(self, s, a, is_agent, code_answer, kl_coef, verbose=False):
        '''
        IN:
        s: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor)
        a: [BATCH_SIZE, COMPRESSED_VOCAB_SIZE](torch.FloatTensor)
        is_agent: [BATCH_SIZE,](torch.FloatTensor)
        code_answer: [BATCH_SIZE,](ndarray)
        OUT:
        loss, kl_loss
        '''
        is_agent = is_agent.view(-1, 1)
        code_answer = torch.as_tensor(code_answer, dtype=torch.long, device=DEVICE)
        dist = self.get_distribution(s, a)
        latent_variable = dist.rsample()
        disc_out = sigmoid(self.disc_out(latent_variable))
        code_out = self.code_out(latent_variable)
        _disc_out = disc_out.detach().cpu().numpy().reshape((-1,))
        if verbose:
            print('disc_out:', -np.tan(_disc_out - 0.5))
        disc_loss = self.disc_loss(disc_out, is_agent)
        code_loss = self.code_loss(code_out, code_answer)
        kl_loss = torch.mean(kl_divergence(dist, self.target_dist))
        loss = disc_loss + WEIGHT_FOR_CODE*code_loss + kl_coef*kl_loss
        kl_loss = kl_loss.detach().cpu().numpy()
        return loss, kl_loss


    def train_by_loss(self, loss):
        self.zero_grad()
        loss.backward()
        self.opt.step()

    def save(self, epoch):
        path = MODEL_SAVEPATH + str(epoch) + "disc" + ".pt"
        torch.save(self.state_dict(), path)

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