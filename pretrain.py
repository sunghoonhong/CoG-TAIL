import codecs
import numpy as np
import matplotlib.pyplot as plt
import torch
import gzip, pickle
from Agent import Agent
from Environment import Environment
from config import *
from util import *
EPOCHS = 200000
TEST_NUM = 100

if __name__ == '__main__':
    with gzip.open('Top5000_dist.pickle') as f:
        dist_dict = pickle.load(f)
    weights = get_weights_from_dict(dist_dict)
    model, tokenizer = get_bert_model_and_tokenizer()
    dist = torch.distributions.Categorical(probs=torch.full((CODE_SIZE,), fill_value=1/CODE_SIZE))
    agent = Agent(model, weights)
#    agent.pretrain_load()
    env = Environment(model, tokenizer)
    loss_list = []
    expert_chunk_generator = get_expert_chunk_generator()
    for epoch in range(EPOCHS):
        expert_chunk = next(expert_chunk_generator)
        loss = agent.pretrain(expert_chunk)
        loss_list.append(loss)
        if epoch % TEST_NUM == 0 and epoch > 0:
            print('epoch: ', epoch, end=' ')
            average_list = moving_average(loss_list)
            plt.plot(np.arange(len(average_list)), np.array(average_list))
            plt.savefig('pretrain_loss.jpg')
            agent.pretrain_save()
            s = env.reset()
            c = dist.sample().numpy().item()
            print('code: ', c)
            d = False
            while not d:
                a, log_prob = agent.get_action(s, c)
                next_s, r, d, _ = env.step(a)
                s = next_s
            if epoch == TEST_NUM:
                f = codecs.open('pretrain_generated_sentence.txt', 'w', "utf-8")
            else:
                f = codecs.open('pretrain_generated_sentence.txt', 'a', 'utf-8')
            sentence = env.id_to_string()
            print(sentence)
            f.write('epoch: ' + str(epoch) + ' code: ' + str(c) + ' sentence: ' + sentence + '\n')
            f.close()
