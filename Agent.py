import numpy as np
import torch
from Actor_Critic import Actor_Critic
from Discriminator import Discriminator
from Memory import ShortMemory, LongMemory
from util import *

class Agent():
    def __init__(self, bert_model, expert_weights=None):
        if expert_weights is not None:
            pretrain_loss_function = torch.nn.CrossEntropyLoss(torch.as_tensor(expert_weights, dtype=torch.float, device=DEVICE))
            self.actor_critic = Actor_Critic(pretrain_loss_function).to(DEVICE)
        else:
            default_weights = np.zeros(VOCAB_SIZE)
            self.actor_critic = Actor_Critic(torch.nn.CrossEntropyLoss(torch.as_tensor(default_weights, dtype=torch.float, device=DEVICE)).to(DEVICE))
        self.discriminator = Discriminator().to(DEVICE)
        self.bert_model = bert_model
        self.short_memory = ShortMemory(self.actor_critic, self.discriminator, bert_model)
        self.long_memory = LongMemory()
        self.horizon_cnt = 0
        

    def get_action(self, state, code):
        '''
        IN:
        state: [STATE_SIZE,](torch.FloatTensor)
        code: single integer
        OUT:
        action: single integer
        log_prob: single float
        '''
        self.actor_critic.eval()
        code = to_onehot(code)
        if len(state.size()) < 2:
            state = state.unsqueeze(0)
        if len(code.size()) < 2:
            code = code.unsqueeze(0)
        action, log_prob = self.actor_critic.action_forward(state, code)
        action = action[0].cpu().numpy().item()
        log_prob = log_prob[0].cpu().numpy().item()
        return action, log_prob

    def store(self, s, a, c, d, next_s, log_prob):
        '''
        IN:
        s: [STATE_SIZE,](torch.FloatTensor)
        a: single integer
        c: single integer
        d: single boolean
        next_s: [STATE_SIZE,](torch.FloatTensor)
        log_prob: single float
        '''
        self.short_memory.append(s, a, c, d, log_prob)
        self.horizon_cnt += 1
        if d or self.horizon_cnt == HORIZON_THRESHOLD:
            self.horizon_cnt = 0
            self.short_memory.set_last_state(next_s)
            self.actor_critic.eval()
            self.discriminator.eval()
            self.short_memory.move_to_long_memory(self.long_memory)
            self.short_memory.flush()
            return self.long_memory.check_update()
        else:
            return False

    def discriminator_update(self, expert_states, expert_actions, expert_codes, verbose=False):
        self.discriminator.train()
        #shuffle expert trajectories
        expert_chunk_length = len(expert_states)
        expert_indice = np.arange(expert_chunk_length)
        np.random.shuffle(expert_indice)
        expert_states = expert_states[expert_indice]
        expert_actions = expert_actions[expert_indice]
        expert_codes = expert_codes[expert_indice]
        #shuffle agent trajectories
        agent_chunk_length = self.long_memory.count
        agent_indice = np.arange(agent_chunk_length)
        np.random.shuffle(agent_indice)
        agent_states = self.long_memory.states[agent_indice]
        agent_actions = self.long_memory.bert_encoded_actions[agent_indice]
        agent_codes = self.long_memory.codes[agent_indice]
        agent_rewards = self.long_memory.rewards[agent_indice]
        for i in range(min(expert_chunk_length//BATCH_SIZE, agent_chunk_length//BATCH_SIZE)):
            batch_agent_states = torch.as_tensor(agent_states[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_agent_actions = torch.as_tensor(agent_actions[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_agent_codes = agent_codes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            is_agent = torch.ones(len(batch_agent_states), dtype=torch.float, device=DEVICE)
            agent_loss = self.discriminator.calculate_loss(batch_agent_states, batch_agent_actions, is_agent, batch_agent_codes, verbose)
            if verbose:
                print('agent_rewards: ', agent_rewards[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            batch_expert_states = torch.as_tensor(expert_states[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_expert_actions = torch.as_tensor(expert_actions[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_expert_codes = expert_codes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            is_agent = torch.zeros(len(batch_expert_states), dtype=torch.float, device=DEVICE)
            expert_loss = self.discriminator.calculate_loss(batch_expert_states, batch_expert_actions, is_agent, batch_expert_codes, False)
            disc_loss = 0.5*expert_loss + 0.5*agent_loss
            print('d loss: ', disc_loss)
            self.discriminator.train_by_loss(disc_loss)

    def actor_critic_update(self):
        self.actor_critic.train()
        #shuffle agent trajectories
        chunk_length = self.long_memory.count
        indice = np.arange(chunk_length)
        np.random.shuffle(indice)
        states = self.long_memory.states[indice]
        actions = self.long_memory.actions[indice]
        codes = self.long_memory.codes[indice]
        gaes = self.long_memory.gaes[indice]
        oracle_values = self.long_memory.oracle_values[indice]
        old_log_probs = self.long_memory.old_log_probs[indice]
        for i in range(chunk_length//BATCH_SIZE):
            batch_states = torch.as_tensor(states[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_actions = torch.as_tensor(actions[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.long, device=DEVICE)
            batch_codes = codes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch_gaes = torch.as_tensor(gaes[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_oracle_values = torch.as_tensor(oracle_values[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_old_log_probs = torch.as_tensor(old_log_probs[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            critic_loss = self.actor_critic.critic_loss(batch_states, batch_codes, batch_oracle_values)
            actor_loss = self.actor_critic.actor_loss(batch_states, batch_actions, batch_codes, batch_gaes, batch_old_log_probs)
            loss = ACTOR_COEF*actor_loss + CRITIC_COEF*critic_loss
            print('a loss: ', actor_loss)
            print('c loss: ', critic_loss)
            self.actor_critic.train_by_loss(loss)


    def update(self, expert_chunk, update_discriminator):
        expert_states = expert_chunk['states']
        expert_actions = expert_chunk['actions']
        expert_codes = expert_chunk['codes'].reshape((-1,))
        tmp_cnt = 0
        if update_discriminator:
            for _ in range(DISC_STEP):
                if tmp_cnt == 0:
                    self.discriminator_update(expert_states, expert_actions, expert_codes, True)
                else:
                    self.discriminator_update(expert_states, expert_actions, expert_codes)
                tmp_cnt += 1
        for _ in range(PPO_STEP):
            self.actor_critic_update()
        self.long_memory.flush()

    def pretrain(self, expert_chunk):
        self.actor_critic.train()
        expert_states = expert_chunk['states']
        expert_action_ids = expert_chunk['action_ids'].reshape((-1,))
        expert_codes = expert_chunk['codes'].reshape((-1,))
        expert_chunk_length = len(expert_states)
        expert_indice = np.arange(expert_chunk_length)
        np.random.shuffle(expert_indice)
        expert_states = expert_states[expert_indice]
        expert_action_ids = expert_action_ids[expert_indice]
        expert_codes = expert_codes[expert_indice]
        loss_sum = 0
        for i in range(expert_chunk_length//BATCH_SIZE):
            batch_expert_states = torch.as_tensor(expert_states[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_expert_action_ids = torch.as_tensor(expert_action_ids[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.long, device=DEVICE)
            batch_expert_codes = expert_codes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            pretrain_loss = self.actor_critic.pretrain_loss(batch_expert_states, batch_expert_action_ids, batch_expert_codes)
            self.actor_critic.pretrain_by_loss(pretrain_loss)
            loss_sum += pretrain_loss.detach().cpu().numpy()
        return loss_sum/(expert_chunk_length//BATCH_SIZE)

    def pretrain_save(self):
        torch.save(self.actor_critic.state_dict(), PRETRAIN_SAVEPATH)

    def pretrain_load(self):
        self.actor_critic.load_state_dict(torch.load(PRETRAIN_SAVEPATH, map_location=torch.device(DEVICE)))
        self.actor_critic.to(DEVICE)
