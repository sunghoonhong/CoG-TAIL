import numpy as np
import torch
from config import *
from util import *


class ShortMemory():
    def __init__(self, actor_critic, discriminator, bert_model):
        self.actor_critic = actor_critic
        self.discriminator = discriminator
        self.bert_model = bert_model
        self.gamma = GAMMA
        self.lambd = LAMBDA
        self.states = []
        self.actions = []
        self.codes = []
        self.rewards = []
        self.last_state = None
        self.done = False
        self.old_log_probs = []

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
        if len(self.states) < 2:
            self.actor_critic.eval()
            self.discriminator.eval()
        bert_encoded_actions = self.actions_to_bert_encoding()
        states_values, last_state_value = self.get_values()
        self.get_rewards(bert_encoded_actions)
        oracle_values, gaes = self.get_gae(states_values, last_state_value)
        states = np.stack(self.states, axis=0)
        bert_encoded_actions = bert_encoded_actions.cpu().numpy()
        long_memory.append(states, self.actions, bert_encoded_actions, self.codes, gaes, oracle_values, self.old_log_probs)

    def actions_to_bert_encoding(self):
        '''
        OUT:
        bert_encoded_actions: [BATCH_SIZE, STATE_SIZE](torch.FloatTensor)
        '''
        tmp = torch.as_tensor(self.actions, dtype=torch.long, device=DEVICE).view(-1, 1)
        with torch.no_grad():
            bert_encoded_actions = self.bert_model.embeddings(tmp).squeeze(1)
        return bert_encoded_actions

    def get_values(self):
        states = torch.as_tensor(np.stack(self.states, axis=0), dtype=torch.float, device=DEVICE)
        last_state = torch.as_tensor(self.last_state, device=DEVICE)
        codes = to_onehot(self.codes)
        code = to_onehot([self.codes[-1]])
        states_values = self.actor_critic.critic_forward(states, codes).detach().cpu().numpy().reshape((-1,))
        self.actor_critic.eval()
        last_state_value = self.actor_critic.critic_forward(last_state, code).detach().cpu().numpy()[0][0]
        return states_values, last_state_value

    def get_rewards(self, bert_encoded_actions):
        states = torch.as_tensor(np.stack(self.states, axis=0), dtype=torch.float, device=DEVICE)
        disc_out, _ = self.discriminator(states, bert_encoded_actions)
        disc_out = disc_out.detach().cpu().numpy().reshape((-1,))
        print('disc_out: ', disc_out)
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
        self.bert_encoded_actions = []
        self.codes = []
        self.gaes = []
        self.oracle_values = []
        self.old_log_probs = []

    def append(self, states, actions, bert_encoded_actions, codes, gaes, oracle_values, old_log_probs):
        '''
        IN:
        states: [HORIZON_LENGTH, STATE_SIZE](ndarray)
        actions: [HORIZON_LENGTH,](list)
        bert_encoded_actions: [HORIZON_LENGTH, STATE_SIZE](ndarray)
        codes: [HORIZON_LENGTH,](list)
        gaes: [HORIZON_LENGTH,](list)
        oracle_values: [HORIZON_LENGTH,](list)
        old_log_probs: [HORIZON_LENGTH,](list)
        '''
        self.count += len(states)
        self.states.append(states)
        self.actions.extend(actions)
        self.bert_encoded_actions.append(bert_encoded_actions)
        self.codes.extend(codes)
        self.gaes.extend(gaes)
        self.oracle_values.extend(oracle_values)
        self.old_log_probs.extend(old_log_probs)

    def flush(self):
        self.count = 0
        self.states = []
        self.actions = []
        self.bert_encoded_actions = []
        self.codes = []
        self.gaes = []
        self.oracle_values = []
        self.old_log_probs = []

    def check_update(self):
        if self.count > THRESHOLD_LEN:
            self.states = np.concatenate(self.states, axis=0)
            self.actions = np.array(self.actions, dtype=np.long)
            self.bert_encoded_actions = np.concatenate(self.bert_encoded_actions, axis=0)
            self.codes = np.array(self.codes, dtype=np.float)
            self.gaes = np.array(self.gaes, dtype=np.float)
            oracle_values = np.array(self.oracle_values, dtype=np.float)
            m = np.mean(oracle_values)
            s = np.std(oracle_values)
            self.oracle_values = (oracle_values - m)/s
            self.old_log_probs = np.array(self.old_log_probs, dtype=np.float)
            return True
        else:
            return False
