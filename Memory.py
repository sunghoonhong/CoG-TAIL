import numpy as np
import torch
import gzip, pickle
from config import *
from util import *


class ShortMemory():
    def __init__(self, actor_critic, discriminator, bert_model, encoder):
        self.actor_critic = actor_critic
        self.discriminator = discriminator
        self.bert_model = bert_model
        self.encoder = encoder
        self.gamma = GAMMA
        self.lambd = LAMBDA
        self.states = []
        self.actions = []
        self.codes = []
        self.rewards = []
        self.last_state = None
        self.done = False
        self.old_log_probs = []
        with gzip.open('Top5000_AtoB.pickle') as f:
            to_bert_dict = pickle.load(f)
        self.to_bert_dict = to_bert_dict

    def append(self, s, a, c, d, log_prob):
        '''
        IN:
        s: [STATE_SIZE,](torch.FloatTensor)
        a: single integer
        c: single integer
        d: single boolean
        log_prob: single float
        '''
        s = s.cpu().numpy()
        self.states.append(s)
        self.actions.append(a)
        self.codes.append(c)
        self.done = d
        self.old_log_probs.append(log_prob)

    def set_last_state(self, s):
        '''
        IN:
        s: [STATE_SIZE,](torch.FloatTensor)
        '''
        s = np.expand_dims(s.cpu().numpy(), axis=0)
        self.last_state = s

    def flush(self):
        self.states = []
        self.actions = []
        self.codes = []
        self.rewards = []
        self.done = False
        self.old_log_probs = []

    def move_to_long_memory(self, long_memory):
        encoded_actions = self.actions_to_encoding()
        states_values, last_state_value = self.get_values()
        self.get_rewards(encoded_actions)
        oracle_values, gaes = self.get_gae(states_values, last_state_value)
        states = np.stack(self.states, axis=0)
        encoded_actions = encoded_actions.cpu().numpy()
        long_memory.append(states, self.actions, encoded_actions, self.codes, gaes, oracle_values, self.old_log_probs, self.rewards)

    def actions_to_encoding(self):
        '''
        OUT:
        encoded_actions: [BATCH_SIZE, COMPRESSED_VOCAB_SIZE](torch.FloatTensor)
        '''
        tmp = []
        for action in self.actions:
            tmp.append(self.to_bert_dict[action])
        tmp = torch.as_tensor(tmp, dtype=torch.long, device=DEVICE).view(-1, 1)
        with torch.no_grad():
            bert_encoded_actions = self.bert_model.embeddings(tmp).squeeze(1)
            encoded_actions = self.encoder.get_latent_variable(bert_encoded_actions)
        return encoded_actions

    def get_values(self):
        states = torch.as_tensor(np.stack(self.states, axis=0), dtype=torch.float, device=DEVICE)
        last_state = torch.as_tensor(self.last_state, device=DEVICE)
        codes = to_onehot(self.codes)
        code = to_onehot([self.codes[-1]])
        states_values = self.actor_critic.critic_forward(states, codes).detach().cpu().numpy().reshape((-1,))
        last_state_value = self.actor_critic.critic_forward(last_state, code).detach().cpu().numpy()[0][0]
        return states_values, last_state_value

    def get_rewards(self, encoded_actions):
        states = torch.as_tensor(np.stack(self.states, axis=0), dtype=torch.float, device=DEVICE)
        disc_out, _ = self.discriminator(states, encoded_actions)
        disc_out = disc_out.detach().cpu().numpy().reshape((-1,))
        #choose your own reward scheme here!
        #log loss with shift
#        self.rewards = list(-np.log(disc_out) + np.log(0.5))
        #log loss
#        self.rewards = list(-np.log(disc_out))
        #tan loss
        self.rewards = list(-np.tan(disc_out - 0.5))
        print('rewards: ',self.rewards)
        #linear loss
#        self.rewards = list(-(disc_out - 0.5))

    def get_gae(self, states_values, last_state_value):
        running_return = 0
        running_adv = 0
        next_value = 0
        returns = np.zeros_like(states_values)
        gaes = np.zeros_like(states_values)
        if self.done == False:
            running_return = last_state_value
            next_value = last_state_value
        for i in reversed(range(len(self.rewards))):
            running_return = self.rewards[i] + self.gamma*running_return
            running_td = self.rewards[i] + self.gamma*next_value - states_values[i]
            running_adv = running_td + self.lambd * self.gamma * running_adv
            returns[i] = running_return
            next_value = states_values[i]
            gaes[i] = running_adv
        return list(returns), list(gaes)

class LongMemory():
    def __init__(self):
        self.count = 0
        self.states = []
        self.actions = []
        self.encoded_actions = []
        self.codes = []
        self.gaes = []
        self.oracle_values = []
        self.old_log_probs = []
        self.rewards = []

    def append(self, states, actions, encoded_actions, codes, gaes, oracle_values, old_log_probs, rewards):
        '''
        IN:
        states: [HORIZON_LENGTH, STATE_SIZE](ndarray)
        actions: [HORIZON_LENGTH,](list)
        encoded_actions: [HORIZON_LENGTH, COMPRESSED_VOCAB_SIZE](ndarray)
        codes: [HORIZON_LENGTH,](list)
        gaes: [HORIZON_LENGTH,](list)
        oracle_values: [HORIZON_LENGTH,](list)
        old_log_probs: [HORIZON_LENGTH,](list)
        '''
        self.count += len(states)
        self.states.append(states)
        self.actions.extend(actions)
        self.encoded_actions.append(encoded_actions)
        self.codes.extend(codes)
        self.gaes.extend(gaes)
        self.oracle_values.extend(oracle_values)
        self.old_log_probs.extend(old_log_probs)
        self.rewards.extend(rewards)

    def flush(self):
        self.count = 0
        self.states = []
        self.actions = []
        self.encoded_actions = []
        self.codes = []
        self.gaes = []
        self.oracle_values = []
        self.old_log_probs = []
        self.rewards = []

    def check_update(self):
        if self.count > THRESHOLD_LEN:
            self.states = np.concatenate(self.states, axis=0)
            self.actions = np.array(self.actions, dtype=np.long)
            self.encoded_actions = np.concatenate(self.encoded_actions, axis=0)
            self.codes = np.array(self.codes, dtype=np.float)
            self.gaes = np.array(self.gaes, dtype=np.float)
            self.oracle_values = np.array(self.oracle_values, dtype=np.float)
            self.old_log_probs = np.array(self.old_log_probs, dtype=np.float)
            self.rewards = np.array(self.rewards, dtype=np.float)
            print('reward avg: ', np.mean(self.rewards))
            REWARD_LIST.append(np.mean(self.rewards))
#            print(len(self.states), len(self.actions), len(self.encoded_actions), len(self.codes), len(self.gaes), len(self.oracle_values), len(self.old_log_probs), len(self.rewards))
            return True
        else:
            return False
