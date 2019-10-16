import torch
import numpy as np
import matplotlib.pyplot as plt
import gzip, pickle
from Autoencoder import Encoder
from ActionEncoder import ActionEncoder
from torch.nn import Module
from torch.nn import Linear, PReLU
from torch.nn import MSELoss
from torch.distributions import MultivariateNormal, kl_divergence
from torch.optim import Adam
from itertools import chain
from config import *
from util import *

def generate_test_data(to_bert_dict, bert_model, encoder:Encoder):
    tmp = []
    indice = np.random.randint(low=0, high=VOCAB_SIZE, size=(AUTOENCODER_BATCH_SIZE))
    for index in indice:
        tmp.append(to_bert_dict[index])
    tmp = torch.as_tensor(tmp, dtype=torch.long, device=DEVICE).view(-1, 1)
    with torch.no_grad():
        bert_encoding = bert_model.embeddings(tmp).squeeze(1)
    #indice to one_hot
    tmp = torch.eye(VOCAB_SIZE)
    tmp = tmp[indice]
    data = torch.as_tensor(tmp, dtype=torch.float, device=DEVICE)
    return bert_encoding.detach() ,data.detach()

if __name__ == '__main__':
    bert_model, _ = get_bert_model_and_tokenizer()
    with gzip.open('Top5000_AtoB.pickle') as f:
        to_bert_dict = pickle.load(f)
    encoder = Encoder(True).to(DEVICE)
    actionencoder = ActionEncoder(True).to(DEVICE)
    for _ in range(100):
        bert_encoding, data = generate_test_data(to_bert_dict, bert_model, encoder)
        d1 = encoder.get_latent_variable(bert_encoding)
        d2 = actionencoder(data)
        print(torch.sum((d1-d2)**2))