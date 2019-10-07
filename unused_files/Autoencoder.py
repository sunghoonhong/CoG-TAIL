import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Module
from torch.nn import Linear
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from itertools import chain
from config import *

class Encoder(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(VOCAB_SIZE, COMPRESSED_VOCAB_SIZE)

    def forward(self, x):
        x = self.linear(x)
        return x

    def save(self):
        torch.save(self.state_dict(), AUTOENCODER_SAVE_PATH)

    def load(self):
        self.load_state_dict(torch.load(AUTOENCODER_SAVE_PATH))



class Decoder(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(COMPRESSED_VOCAB_SIZE, VOCAB_SIZE)

    def forward(self, x):
        x = self.linear(x)
        return x

def generate_data():
    indice = np.random.randint(low=0, high=VOCAB_SIZE, size=(AUTOENCODER_BATCH_SIZE))
    data = np.zeros((AUTOENCODER_BATCH_SIZE, VOCAB_SIZE))
    data[np.arange(AUTOENCODER_BATCH_SIZE), indice] = 1
    data = torch.FloatTensor(data)
    answer = torch.LongTensor(indice)
    return data, answer

if __name__ == '__main__':
    train = False
    if train:
        ITERATION = 50000
        encoder = Encoder().cuda()
        decoder = Decoder().cuda()
        parameters = chain(encoder.parameters(), decoder.parameters())
        opt = Adam(parameters, lr=1e-5)
        cross_entropy_loss = CrossEntropyLoss()
        loss_list = []

        for i in range(ITERATION):
            original, answer = generate_data()
            original = original.cuda()
            answer = answer.cuda()
            processed = decoder(encoder(original))
            loss = cross_entropy_loss(processed, answer)
            loss.backward()
            opt.step()
            loss_list.append(loss.to('cpu').item())
            if i % 500 == 0 and i > 0:
                plt.plot(np.arange(len(loss_list)), loss_list)
                plt.savefig('autoencoder_loss.jpg')
                _, indice = torch.max(processed, 1)
                print(torch.eq(answer, indice))
                process_sorted, _ = torch.sort(processed, descending=True)
                print(process_sorted.to('cpu'))
                encoder.save()
    else:
        encoder = Encoder().cuda()
        encoder.load()
        original, answer = generate_data()
        original = original.cuda()
        answer = answer.cuda()
        print(encoder(original))


