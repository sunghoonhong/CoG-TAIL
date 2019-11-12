import numpy as np
import torch
from torch.nn import Module, ModuleList
from torch.nn import Linear, LeakyReLU, PReLU, BatchNorm1d, Bilinear
from torch.nn import BCELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.distributions import MultivariateNormal, kl_divergence
from torch import sigmoid, tanh
from torch.optim import Adam
from util import *
from config import *

"""
input: (s, a) pair
output: score of reality, code reconstruction
"""

class Discriminator(Module):
    def __init__(self):
        super().__init__()
        self.bilinear = Bilinear(STATE_SIZE, COMPRESSED_VOCAB_SIZE, DISC_HIDDEN_UNIT_NUM)
        self.a1 = PReLU()
        self.l2 = Linear(DISC_HIDDEN_UNIT_NUM, AUTOENCODER_HIDDEN_UNIT_NUM)
        self.a2 = PReLU()
        self.l3 = Linear(AUTOENCODER_HIDDEN_UNIT_NUM, DISC_LATENT_SIZE)
        self.disc_out = Linear(DISC_LATENT_SIZE, 1)
        self.code_out = Linear(DISC_LATENT_SIZE, CODE_SIZE)
        self.disc_loss = BCELoss()
        self.code_loss = CrossEntropyLoss()
        self.target_mean = torch.zeros(1, DISC_LATENT_SIZE).to(DEVICE)
        self.target_cov = torch.diag_embed(torch.full((1, DISC_LATENT_SIZE), 0.1)).to(DEVICE)
        self.target_dist = MultivariateNormal(self.target_mean, self.target_cov)
        self.opt = Adam(self.parameters(), lr=DISC_LR, weight_decay=1e-4)

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
        #no sampling
        s_a = self.bilinear(s, a)
        x = self.a1(s_a)
        x = self.l3(self.a2(self.l2(x)))
        m = x[:, :DISC_LATENT_SIZE]
        cov = self.target_cov
        dist = MultivariateNormal(m, cov)
        latent_variable = dist.mean
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
        s_a = self.bilinear(s, a)
        x = self.a1(s_a)
        x = self.l3(self.a2(self.l2(x)))
        m = x[:, :DISC_LATENT_SIZE]
        cov = self.target_cov
        dist = MultivariateNormal(m, cov)
        return dist

    def calculate_loss_with_code(self, s, a, is_agent, code_answer, kl_coef, verbose=True):
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
        latent_variable = dist.mean
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

    def calculate_loss_without_code(self, s, a, is_agent, kl_coef, verbose=False):
        '''
        IN:
        s: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor)
        a: [BATCH_SIZE, COMPRESSED_VOCAB_SIZE](torch.FloatTensor)
        is_agent: [BATCH_SIZE,](torch.FloatTensor)
        OUT:
        loss, kl_loss
        '''
        is_agent = is_agent.view(-1, 1)
        dist = self.get_distribution(s, a)
        latent_variable = dist.mean
        disc_out = sigmoid(self.disc_out(latent_variable))
        _disc_out = disc_out.detach().cpu().numpy().reshape((-1,))
        if verbose:
            print('disc_out:', -np.tan(_disc_out - 0.5))
        disc_loss = self.disc_loss(disc_out, is_agent)
        kl_loss = torch.mean(kl_divergence(dist, self.target_dist))
        loss = disc_loss + kl_coef*kl_loss
        kl_loss = kl_loss.detach().cpu().numpy()
        return loss, kl_loss

    def calculate_vail_loss(self, s, a, code_answer, kl_coef, verbose=True):
        '''
        IN:
        s: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor), first half is agent, last half is expert
        a: [BATCH_SIZE, COMPRESSED_VOCAB_SIZE](torch.FloatTensor)
        code_answer: [BATCH_SIZE,](ndarray)
        '''
        half_batch_size = int(len(s)/2)
        code_answer = torch.as_tensor(code_answer, dtype=torch.long, device=DEVICE)[half_batch_size:]
        dist = self.get_distribution(s, a)
        latent_variable = dist.rsample()
        #calculate disc_loss
        disc_out = sigmoid(self.disc_out(latent_variable).view(-1,))
        agent_out = disc_out[:half_batch_size]
        expert_out = disc_out[half_batch_size:]
        zeros = torch.zeros(half_batch_size).to(DEVICE)
        ones = torch.ones(half_batch_size).to(DEVICE)
        assert len(agent_out) == len(expert_out)
        agent_loss = self.disc_loss(agent_out, zeros)
        expert_loss = self.disc_loss(expert_out, ones)
        disc_loss = (1/2)*(agent_loss + expert_loss)
        #to debug
        _disc_out = disc_out.detach().cpu().numpy().reshape((-1,))
        if verbose:
            print('disc_out:', _disc_out)
        #calculate code_loss
        code_out = self.code_out(latent_variable)[half_batch_size:]
        code_loss = self.code_loss(code_out, code_answer)
        #calculate regularization term
        '''
        s_a = torch.cat((s, a), 1)
        agent_in = s_a[:half_batch_size]
        expert_in = s_a[half_batch_size:]
        epsilon = torch.rand(half_batch_size, 1).expand(half_batch_size, STATE_SIZE + COMPRESSED_VOCAB_SIZE).to(DEVICE)
        x_hat = epsilon*agent_in + (1-epsilon)*expert_in
        x_hat.requires_grad_()
        x = self.a1(self.l1(x_hat))
        x = self.l3(self.a2(self.l2(x)))
        m = x[:, :DISC_LATENT_SIZE]
        D_x_hat = self.disc_out(m)
        grads = torch.autograd.grad(D_x_hat, x_hat,
            grad_outputs=torch.ones(D_x_hat.size()).to(DEVICE),
            retain_graph=True, create_graph=True)[0]
        grads = grads.view(grads.size(0), -1)
        reg_term = torch.mean((grads.norm(2, dim=1) - 1)**2)
        '''
        '''
        d = euclidean(expert_in, agent_in)
        if torch.mean(expert_out - agent_out) < LIPSCHITZ*torch.mean(d):
            reg_term = 0
        else:
        reg_term = torch.mean((1/(4*WAIL_EPSILON)) * ((expert_out - agent_out - LIPSCHITZ*d)**2))
        '''
        #kl loss
        kl_loss = torch.mean(kl_divergence(dist, self.target_dist))
        #loss sum up
        if verbose:
            print('disc_loss: ', disc_loss, ' WEIGHT_FOR_CODE*code_loss: ', WEIGHT_FOR_CODE*code_loss, ' kl_coef*kl: ', kl_coef*kl_loss)
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