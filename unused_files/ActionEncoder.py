import torch
import numpy as np
import matplotlib.pyplot as plt
import gzip, pickle
from Autoencoder import Encoder
from torch.nn import Module
from torch.nn import Linear, PReLU
from torch.nn import MSELoss
from torch.distributions import MultivariateNormal, kl_divergence
from torch.optim import Adam
from itertools import chain
from config import *
from util import *

ITERATION = 100000

class ActionEncoder(Module):
    def __init__(self, load_mymodel):
        super().__init__()
        self.linear = Linear(VOCAB_SIZE, COMPRESSED_VOCAB_SIZE)
        if load_mymodel:
            self.load()

    def forward(self, x):
        return self.linear(x)

    def save(self):
        torch.save(self.state_dict(), ACTIONENCODER_SAVE_PATH)

    def load(self):
        self.load_state_dict(torch.load(ACTIONENCODER_SAVE_PATH, map_location=torch.device(DEVICE)))
        self.to(DEVICE)

# data: 3600차원 랜덤 인덱스 / answer: 인덱스들의 768차원 버트 임베딩을 오토인코더에 넣은 값
def generate_data(to_bert_dict, bert_model, encoder:Encoder):
    tmp = []
    indice = np.random.randint(low=0, high=VOCAB_SIZE, size=(AUTOENCODER_BATCH_SIZE))
    for index in indice:
        tmp.append(to_bert_dict[index])
    tmp = torch.as_tensor(tmp, dtype=torch.long, device=DEVICE).view(-1, 1)
    with torch.no_grad():
        answer = bert_model.embeddings(tmp).squeeze(1)
        answer = encoder.get_latent_variable(answer)
    answer = answer.to(DEVICE)
    #indice to one_hot
    tmp = torch.eye(VOCAB_SIZE)
    tmp = tmp[indice]
    data = torch.as_tensor(tmp, dtype=torch.float, device=DEVICE)
    return data.detach(), answer.detach()


if __name__ == '__main__':
    bert_model, _ = get_bert_model_and_tokenizer()
    # Top3600 인덱스에서 실제 버트 vocab 인덱스로 변환
    with gzip.open('Top3600_AtoB.pickle') as f:
        to_bert_dict = pickle.load(f)
    encoder = Encoder(True).to(DEVICE)
    action_encoder = ActionEncoder(False).to(DEVICE)
    opt = Adam(action_encoder.parameters(), lr=1e-3)
    loss_function = MSELoss()
    for e in range(ITERATION):
        data, answer = generate_data(to_bert_dict, bert_model, encoder)
        x = action_encoder(data)
        loss = loss_function(x, answer)
        action_encoder.zero_grad()
        loss.backward()
        opt.step()
        print(e, ':',loss.detach())
        if e % 100 == 0:
            action_encoder.save()
