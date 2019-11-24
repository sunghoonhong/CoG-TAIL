import codecs
import torch
import gzip, pickle
import matplotlib.pyplot as plt
from Environment import Environment
from Agent import Agent
from config import *
from util import *

MAX_LEN = 20

def preprocess(review: list, total :int, show_progress: bool = True) -> list:
    global counter
    counter += 1
    if counter % 10000 == 0:
        print('Processing... %6i/%6i'% (counter, total))

    states = []
    actions = []
    
    for i in range(2, len(review)):
        states.append(torch.Tensor(review[1:i]+[0]*(MAX_LEN-i+1)))
        actions.append(torch.Tensor([review[i]]))
    
    states = torch.stack(states)
    actions = torch.stack(actions)

    return states, actions

if __name__ == '__main__':
    assert PPO_STEP >= DISC_STEP
    load_model = False
    with gzip.open('Top5000_dist.pickle') as f:
        dist_dict = pickle.load(f)
    with gzip.open('Top5000_first_pos.pickle') as f:
        pos_first_list = pickle.load(f)
    with gzip.open('Top5000_first_neg.pickle') as f:
        neg_first_list = pickle.load(f)
    weights = get_weights_from_dict(dist_dict)
    env = Environment(pos_first_list, neg_first_list)
    agent = Agent(weights)
    agent.pretrain_load()
    actor_update_cnt = 0
    dist = torch.distributions.Categorical(probs=torch.full((CODE_SIZE,), fill_value=1/CODE_SIZE))
    pretrain_loss_list = []
    actor_update_cnt = 0

    counter = 0

    states = []
    actions = []
    codes = []

    cur_size = 0
    batch_size = 512
    num = 1

    for e in range(400000):

        c = dist.sample().numpy().item()
        s = env.reset(c)
        d = False
        while not d:
            a, _ = agent.get_action(s, c, test=True)
            next_s, _, d, _ = env.step(a, test=True)
            s = next_s
        sentence = env.sentence

        parts = preprocess(sentence, 400000)

        states.append(parts[0])
        actions.append(parts[1])
        codes.append(torch.full((len(parts[0]), 1), c).long())
    
        cur_size += len(parts[0])
        
        if cur_size >= batch_size:
            np.savez_compressed('./bpe_fake_token/Amazon_Dataset'+str(num),
                                states = torch.cat(states, dim=0),
                                actions = torch.cat(actions, dim=0),
                                codes = torch.cat(codes, dim=0))

            states = []
            actions = []
            codes = []
            action_ids = []
            prev_action_ids = []
            
            cur_size = 0
            num += 1