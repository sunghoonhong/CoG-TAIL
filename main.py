import codecs
import torch
import gzip, pickle
import matplotlib.pyplot as plt
from Environment import Environment
from Agent import Agent
from Autoencoder import Encoder
from config import *
from util import *

if __name__ == '__main__':
    assert PPO_STEP > DISC_STEP
    with gzip.open('Top5000_dist.pickle') as f:
        dist_dict = pickle.load(f)
    with gzip.open('Top5000_first_list.pickle') as f:
        first_list = pickle.load(f)
    weights = get_weights_from_dict(dist_dict)
    bert_model, bert_tokenizer = get_bert_model_and_tokenizer()
    env = Environment(bert_model, bert_tokenizer, first_list)
    agent = Agent(bert_model, weights)
#    agent.pretrain_load()
    expert_chunk_generator = get_expert_chunk_generator()
    dist = torch.distributions.Categorical(probs=torch.full((CODE_SIZE,), fill_value=1/CODE_SIZE))
    disc_update_cnt = DISC_UPDATE_CNT
    pretrain_loss_list = []
    for e in range(1000000):
        s = env.reset()
        c = dist.sample().numpy().item()
        d = False
        while not d:
            a, log_prob = agent.get_action(s, c)
            next_s, r, d, _ = env.step(a)
            time_to_update = agent.store(s, a, c, d, next_s, log_prob)
            if time_to_update:
                if disc_update_cnt == DISC_UPDATE_CNT:
                    update_discriminator = True
                    disc_update_cnt = 0
                else:
                    update_discriminator = False
                    disc_update_cnt += 1
                print('updating')
                expert_chunks = get_n_expert_batch(expert_chunk_generator, PPO_STEP)
                pretrain_loss = agent.update(expert_chunks, update_discriminator)
                pretrain_loss_list.append(pretrain_loss)
                print('update complete')
            s = next_s
#        print(env.sentence)
        if e % TRAIN_REPORT_PERIOD == 0:
            if len(pretrain_loss_list) > MOVING_AVERAGE:
                average_list = moving_average(pretrain_loss_list)
                plt.plot(np.arange(len(average_list)), np.array(average_list))
                plt.savefig('_pretrain_loss.jpg')
            if e == 0:
                f = codecs.open('train_generated_sentence.txt', 'w', "utf-8")
            else:
                f = codecs.open('train_generated_sentence.txt', 'a', 'utf-8')
            sentence = env.id_to_string()
            print(sentence)
            f.write('epoch: ' + str(e) + ' code: ' + str(c) + ' sentence: ' + sentence + '\n')
            f.close()


