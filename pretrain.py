import codecs
import numpy as np
import matplotlib.pyplot as plt
import torch
import gzip, pickle
from Agent import Agent
from Environment import Environment
from config import *
from util import *
FILE_NUM = 5115
EPOCH = 20
TEST_NUM = 100

if __name__ == '__main__':
    with gzip.open('Top3600_dist.pickle') as f:
        dist_dict = pickle.load(f)
    with gzip.open('Top3600_first_pos.pickle') as f:
        pos_first_list = pickle.load(f)
    with gzip.open('Top3600_first_neg.pickle') as f:
        neg_first_list = pickle.load(f)     
    weights = get_weights_from_dict(dist_dict)
    # model, tokenizer = get_bert_model_and_tokenizer()
    tokenizer = get_tokenizer()
    dist = torch.distributions.Categorical(probs=torch.full((CODE_SIZE,), fill_value=1/CODE_SIZE))
    agent = Agent(weights)
#    agent.pretrain_load()
    env = Environment(pos_first_list, neg_first_list)
    loss_list = []
    expert_chunk_generator = get_expert_chunk_generator()
    for epoch in range(FILE_NUM*EPOCH):
        expert_chunk = next(expert_chunk_generator)
        loss = agent.pretrain(expert_chunk)
        loss_list.append(loss)
        if epoch % TEST_NUM == 0 and epoch > 0:
            print('epoch: ', epoch, end=' ')
            average_list = moving_average(loss_list)
            if len(average_list) > 50:
                plt.plot(np.arange(len(average_list)), np.array(average_list))
            plt.savefig('pretrain_loss.jpg')
            agent.pretrain_save()
            c = dist.sample().numpy().item()
            s = env.reset(c)
            print('code: ', c)
            d = False
            while not d:
                a, log_prob = agent.get_action(s, c, test=True)
                next_s, r, d, _ = env.step(a, test=True)
                s = next_s
            if epoch == TEST_NUM:
                f = codecs.open('pretrain_generated_sentence.txt', 'w', "utf-8")
            else:
                f = codecs.open('pretrain_generated_sentence.txt', 'a', 'utf-8')
            sentence = env.id_to_string()
            print(sentence)
            f.write('epoch: ' + str(epoch) + ' code: ' + str(c) + ' sentence: ' + sentence + '\n')
            f.close()
