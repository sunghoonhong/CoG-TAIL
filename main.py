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
    load_model = False
    load_epoch = 105000
    with gzip.open('Top3600_dist.pickle') as f:
        dist_dict = pickle.load(f)
    with gzip.open('Top3600_first_pos.pickle') as f:
        pos_first_list = pickle.load(f)
    with gzip.open('Top3600_first_neg.pickle') as f:
        neg_first_list = pickle.load(f)
    weights = get_weights_from_dict(dist_dict)
    bert_model, bert_tokenizer = get_bert_model_and_tokenizer()
    env = Environment(bert_model, bert_tokenizer, pos_first_list, neg_first_list)
    agent = Agent(bert_model, weights)
    if load_model == False:
        agent.pretrain_load()
        disc_update_cnt = DISC_UPDATE_CNT
    else:
        agent.load(load_epoch)
        disc_update_cnt = AFTER_TRAIN_DISC_UPDATE_CNT
    expert_chunk_generator = get_expert_chunk_generator()
    dist = torch.distributions.Categorical(probs=torch.full((CODE_SIZE,), fill_value=1/CODE_SIZE))
    pretrain_loss_list = []
    _disc_update_cnt = AFTER_TRAIN_DISC_UPDATE_CNT
    for e in range(1000000):
        if load_model:
            e += load_epoch
        c = dist.sample().numpy().item()
        s = env.reset(c)
        d = False
        while not d:
            a, log_prob = agent.get_action(s, c)
            next_s, r, d, _ = env.step(a)
            time_to_update = agent.store(s, a, c, d, next_s, log_prob)
            if time_to_update:
                if _disc_update_cnt == disc_update_cnt:
                    update_discriminator = True
                    _disc_update_cnt = 0
                else:
                    update_discriminator = False
                    _disc_update_cnt += 1
                print('updating')
                expert_chunks = get_n_expert_batch(expert_chunk_generator, PPO_STEP)
                pretrain_loss = agent.update(expert_chunks, update_discriminator)
                pretrain_loss_list.append(pretrain_loss)
                print('update complete')
            s = next_s
#        print(env.sentence)
        if e % TRAIN_REPORT_PERIOD == 0:
            if len(REWARD_LIST) > 1:
                plt.plot(np.arange(len(REWARD_LIST)), np.array(REWARD_LIST))
                plt.savefig('reward_list.jpg')
            if e == 0:
                f = codecs.open('train_generated_sentence.txt', 'w', "utf-8")
            else:
                f = codecs.open('train_generated_sentence.txt', 'a', 'utf-8')
            
            c = dist.sample().numpy().item()
            s = env.reset(c)
            d = False
            while not d:
                a, _ = agent.get_action(s, c, test=True)
                next_s, _, d, _ = env.step(a, test=True)
                s = next_s
            sentence = env.id_to_string()
            print(sentence)
            f.write('epoch: ' + str(e) + ' code: ' + str(c) + ' sentence: ' + sentence + '\n')
            f.close()
        if e % MODEL_SAVE_PERIOD == 0:
            agent.save(e)


