import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from Actor_Critic import Actor_Critic
from Discriminator import Discriminator
from Autoencoder import Encoder
from ActionEncoder import ActionEncoder
from Memory import ShortMemory, LongMemory
from util import *

class Agent():
    def __init__(self, bert_model, expert_weights=None):
        if expert_weights is not None:
            pretrain_loss_function = CrossEntropyLoss(torch.as_tensor(expert_weights, dtype=torch.float, device=DEVICE))
            self.actor_critic = Actor_Critic(pretrain_loss_function).to(DEVICE)
        else:
            default_weights = np.zeros(VOCAB_SIZE)
            self.actor_critic = Actor_Critic(CrossEntropyLoss(torch.as_tensor(default_weights, dtype=torch.float, device=DEVICE)).to(DEVICE))
        self.discriminator = Discriminator().to(DEVICE)
        self.bert_model = bert_model
        self.encoder = Encoder(True)
        self.actionencoder = ActionEncoder(True)
        self.short_memory = ShortMemory(self.actor_critic, self.discriminator, bert_model, self.encoder)
        self.long_memory = LongMemory()
        self.info_loss_function = CrossEntropyLoss()
        self.horizon_cnt = 0
        self.kl_coef = 0.1
        

    def get_action(self, state, code, test=False):
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
        action, log_prob = self.actor_critic.action_forward(state, code, test=test)
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

    def discriminator_update(self, expert_states, expert_actions, expert_codes):
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
        agent_actions = self.long_memory.encoded_actions[agent_indice]
#        agent_codes = self.long_memory.codes[agent_indice]
        for i in range(min(expert_chunk_length//BATCH_SIZE, agent_chunk_length//BATCH_SIZE)):
            #agent forward
            batch_agent_states = torch.as_tensor(agent_states[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_agent_actions = torch.as_tensor(agent_actions[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
#            batch_agent_codes = agent_codes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            is_agent = torch.ones(len(batch_agent_states), dtype=torch.float, device=DEVICE)
            agent_loss, kl_agent = self.discriminator.calculate_loss_without_code(batch_agent_states, batch_agent_actions, is_agent, self.kl_coef)
            #expert forward
            batch_expert_states = torch.as_tensor(expert_states[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_expert_actions = torch.as_tensor(expert_actions[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_expert_actions = self.encoder.get_latent_variable(batch_expert_actions)
            batch_expert_codes = expert_codes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            is_agent = torch.zeros(len(batch_expert_states), dtype=torch.float, device=DEVICE)
            expert_loss, kl_expert = self.discriminator.calculate_loss_with_code(batch_expert_states, batch_expert_actions, is_agent, batch_expert_codes, self.kl_coef)
            kl = 0.5*kl_agent + 0.5*kl_expert
            self.kl_coef = max(0, self.kl_coef + KL_STEP*(kl - IC))
            disc_loss = 0.5*expert_loss + 0.5*agent_loss
            print('d loss: ', disc_loss, ' self.kl_coef: ', self.kl_coef)
            self.discriminator.train_by_loss(disc_loss)

    def actor_critic_update(self, expert_states, expert_action_ids, expert_codes):
        self.actor_critic.train()
        #shuffle agent trajectories
        agent_chunk_length = self.long_memory.count
        indice = np.arange(agent_chunk_length)
        np.random.shuffle(indice)
        states = self.long_memory.states[indice]
        actions = self.long_memory.actions[indice]
        codes = self.long_memory.codes[indice]
        gaes = self.long_memory.gaes[indice]
        oracle_values = self.long_memory.oracle_values[indice]
        old_log_probs = self.long_memory.old_log_probs[indice]
        #shuffle expert trajectories
        expert_chunk_length = len(expert_states)
        expert_indice = np.arange(expert_chunk_length)
        np.random.shuffle(expert_indice)
        expert_states = expert_states[expert_indice]
        expert_action_ids = expert_action_ids[expert_indice]
        expert_codes = expert_codes[expert_indice]
        pretrain_loss_sum = 0
        for i in range(min(expert_chunk_length//BATCH_SIZE, agent_chunk_length//BATCH_SIZE)):
            #pretrain loss
            batch_expert_states = torch.as_tensor(expert_states[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_expert_action_ids = torch.as_tensor(expert_action_ids[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.long, device=DEVICE)
            batch_expert_codes = expert_codes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            pretrain_loss = self.actor_critic.pretrain_loss(batch_expert_states, batch_expert_action_ids, batch_expert_codes)
            pretrain_loss_sum += pretrain_loss.detach().cpu().numpy()
            #actor critic loss
            batch_states = torch.as_tensor(states[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_actions = torch.as_tensor(actions[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.long, device=DEVICE)
            batch_codes = codes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch_gaes = torch.as_tensor(gaes[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_oracle_values = torch.as_tensor(oracle_values[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            batch_old_log_probs = torch.as_tensor(old_log_probs[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=torch.float, device=DEVICE)
            critic_loss = self.actor_critic.critic_loss(batch_states, batch_codes, batch_oracle_values)
            actor_loss = self.actor_critic.actor_loss(batch_states, batch_actions, batch_codes, batch_gaes, batch_old_log_probs)
            #info loss
            one_hot_codes = to_onehot(batch_codes)
            action_logits = self.actor_critic(batch_states, one_hot_codes)
            one_hot_action = gumbel_softmax(action_logits)
            encoded_action = self.actionencoder(one_hot_action)
            _, code_out = self.discriminator(batch_states, encoded_action)
            info_loss = self.info_loss_function(code_out, torch.as_tensor(batch_codes, dtype=torch.long, device=DEVICE))
            #integrated loss
            loss = PRETRAIN_COEF*pretrain_loss + ACTOR_COEF*actor_loss + CRITIC_COEF*critic_loss + INFO_COEF*info_loss
            print('a loss: ', actor_loss, end=' ')
            print('c loss: ', critic_loss, end=' ')
            print('p loss: ', pretrain_loss, end=' ')
            print('info loss: ', info_loss)
            self.actor_critic.train_by_loss(loss)
        return pretrain_loss_sum/(min(expert_chunk_length//BATCH_SIZE, agent_chunk_length//BATCH_SIZE))


    def update(self, expert_chunks, update_discriminator):
        '''
        updates policy, and selectively discriminator
        IN:
        expert_chunks: list of expert_chunk, length: PPO_STEP
        update_discriminator: boolean type flag which determines whether to update discriminator or not
        '''
        pretrain_loss = 0
        if update_discriminator:
            for i in range(DISC_STEP):
                expert_chunk = expert_chunks[i]
                expert_states = expert_chunk['states']
                expert_actions = expert_chunk['actions']
                expert_codes = expert_chunk['codes'].reshape((-1,))
                self.discriminator_update(expert_states, expert_actions, expert_codes)
        for i in range(PPO_STEP):
            expert_chunk = expert_chunks[i]
            expert_states = expert_chunk['states']
            expert_action_ids = expert_chunk['action_ids'].reshape((-1,))
            expert_codes = expert_chunk['codes'].reshape((-1,))
            pretrain_loss += self.actor_critic_update(expert_states, expert_action_ids, expert_codes)
        self.long_memory.flush()
        return pretrain_loss/PPO_STEP

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

    def save(self, epoch):
        epoch += 55000
        self.actor_critic.save(epoch)
        self.discriminator.save(epoch)

    def load(self, epoch):
        self.actor_critic.load_state_dict(torch.load(MODEL_SAVEPATH + str(epoch) + 'ac.pt', map_location=torch.device(DEVICE)))
        self.actor_critic.to(DEVICE)
        self.discriminator.load_state_dict(torch.load(MODEL_SAVEPATH + str(epoch) + 'disc.pt', map_location=torch.device(DEVICE)))
        self.discriminator.to(DEVICE)
