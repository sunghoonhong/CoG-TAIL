import numpy as np
import matplotlib.pyplot as plt
import torch
from Agent import Agent
from Environment import Environment
from config import *
from util import *
EPOCHS = 10000
TEST_NUM = 10

if __name__ == '__main__':
    model, tokenizer = get_bert_model_and_tokenizer()
    dist = torch.distributions.Categorical(probs=torch.full((CODE_SIZE,), fill_value=1/CODE_SIZE))
    agent = Agent(model)
    env = Environment(model, tokenizer)
    loss_list = []
    expert_chunk_generator = get_expert_chunk_generator()
    for epoch in range(EPOCHS):
        print('epoch: ', epoch)
        expert_chunk = next(expert_chunk_generator)
        loss = agent.pretrain(expert_chunk)
        loss_list.append(loss)
        if epoch % TEST_NUM == 0 and epoch > 0:
            plt.plot(np.arange(len(loss_list)), np.array(loss_list))
            plt.savefig('asdf.jpg')
            s = env.reset()
            c = dist.sample().numpy().item()
            d = False
            while not d:
                a, log_prob = agent.get_action(s, c)
                print(a, end=' ')
                next_s, r, d, _ = env.step(a)
                s = next_s
            print()
            print(env.sentence)
            print(env.id_to_string())