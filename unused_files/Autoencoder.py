import torch
import numpy as np
import matplotlib.pyplot as plt
import gzip, pickle
from torch.nn import Module
from torch.nn import Linear, PReLU
from torch.nn import CrossEntropyLoss
from torch.distributions import MultivariateNormal, kl_divergence
from torch.optim import Adam
from itertools import chain
from config import *
from util import *

class Encoder(Module):
    def __init__(self, load_mymodel=False):
        super().__init__()
        self.linear = Linear(STATE_SIZE, AUTOENCODER_HIDDEN_UNIT_NUM)
        self.prelu = PReLU()
        self.out_layer = Linear(AUTOENCODER_HIDDEN_UNIT_NUM, 2*COMPRESSED_VOCAB_SIZE)
        if load_mymodel:
            self.load(11500)

    def forward(self, x):
        x = self.out_layer(self.prelu(self.linear(x)))
        m = x[:, :COMPRESSED_VOCAB_SIZE]
        log_s = x[:, COMPRESSED_VOCAB_SIZE:]
        return m, log_s

    def get_latent_variable(self, x):
        x = self.out_layer(self.prelu(self.linear(x)))
        m = x[:, :COMPRESSED_VOCAB_SIZE]
        return m

    def save(self, epoch):
        path = AUTOENCODER_SAVE_PATH + str(epoch) + ".pt"
        torch.save(self.state_dict(), path)

    def load(self, epoch):
        self.load_state_dict(torch.load(AUTOENCODER_SAVE_PATH + str(epoch) + ".pt", map_location=torch.device(DEVICE)))
        self.to(DEVICE)



class Decoder(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(COMPRESSED_VOCAB_SIZE, VOCAB_SIZE)

    def forward(self, x):
        x = self.linear(x)
        return x

def generate_data(to_bert_dict, bert_model):
    tmp = []
    indice = np.random.randint(low=0, high=VOCAB_SIZE, size=(AUTOENCODER_BATCH_SIZE))
    for index in indice:
        tmp.append(to_bert_dict[index])
    tmp = torch.as_tensor(tmp, dtype=torch.long, device=DEVICE).view(-1, 1)
    with torch.no_grad():
        data = bert_model.embeddings(tmp).squeeze(1)
    data = data.to(DEVICE)
    answer = torch.as_tensor(indice, dtype=torch.long, device=DEVICE)
    return data, answer

if __name__ == '__main__':
    bert_model, _ = get_bert_model_and_tokenizer()
    with gzip.open('Top3600_AtoB.pickle') as f:
        to_bert_dict = pickle.load(f)
    train = True
    if train:
        ITERATION = 12000
        encoder = Encoder().to(DEVICE)
        decoder = Decoder().to(DEVICE)
        target_mean = torch.zeros(AUTOENCODER_BATCH_SIZE, COMPRESSED_VOCAB_SIZE).to(DEVICE)
        target_cov = torch.diag_embed(torch.ones(AUTOENCODER_BATCH_SIZE, COMPRESSED_VOCAB_SIZE)).to(DEVICE)
        target_dist = MultivariateNormal(target_mean, target_cov)
        parameters = chain(encoder.parameters(), decoder.parameters())
        opt = Adam(parameters, lr=1e-3)
        loss_function = CrossEntropyLoss()
        save_epoch = [3000, 5000, 70000, 9000, 10000, 11000, 11500]
        loss_list = []
        answer_cnt_list = []

        for i in range(ITERATION):
            data, answer = generate_data(to_bert_dict, bert_model)
            m, log_s = encoder(data)
            s = torch.exp(log_s)
            dist = MultivariateNormal(m, torch.diag_embed(s))
            latent_variable = dist.rsample()
            processed = decoder(latent_variable)
            cross_entropy_loss = loss_function(processed, answer)
            kl_loss = torch.mean(kl_divergence(dist, target_dist))
            loss = cross_entropy_loss + AUTOENCODER_KL_COEF*kl_loss
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            opt.step()
            loss_list.append(loss.to('cpu').item())
            if i % 500 == 0 and i > 0:
                avg_list = moving_average(loss_list)
                plt.subplot(211)
                plt.plot(np.arange(len(avg_list)), avg_list)
                if i in save_epoch:
                    encoder.save(i)
                inferred = torch.argmax(processed, dim=1)
                ones = torch.ones(AUTOENCODER_BATCH_SIZE).to(DEVICE)
                zeros = torch.zeros(AUTOENCODER_BATCH_SIZE).to(DEVICE)
                answersheet = torch.where(answer == inferred, ones, zeros)
                answersheet = answersheet.detach().cpu().numpy()
                answer_cnt = 0
                for i in answersheet:
                    if i == 1.0:
                        answer_cnt += 1
                answer_cnt = answer_cnt/AUTOENCODER_BATCH_SIZE
                print('answer rate: ', answer_cnt)
                answer_cnt_list.append(answer_cnt)
                plt.subplot(212)
                plt.plot(np.arange(len(answer_cnt_list)), answer_cnt_list)
                plt.savefig('autoencoder_stat.jpg')

    else:
        encoder = Encoder().cuda()
        encoder.load()
        data = generate_data(to_bert_dict,bert_model)
        print(encoder(data))


