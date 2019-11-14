import torch
import random
import gzip, pickle
from transformers import BertTokenizer, BertModel
from copy import copy
from config import *
from util import *
from Agent import Agent

class Environment():
    def __init__(self, bert_model, bert_tokenizer, pos_first_list, neg_first_list):
        self.bert_model = bert_model
        self.tokenizer = bert_tokenizer
        self.pos_first_list = pos_first_list
        self.neg_first_list = neg_first_list
        self.vocab_size = VOCAB_SIZE
        self.observation_space = STATE_SIZE
        with gzip.open('Top3600_AtoB.pickle') as f:
            to_bert_dict = pickle.load(f)
        self.to_bert_dict = to_bert_dict
    
    def reset(self, c):
        '''
        IN:
        c: code
        OUT:
        obs: [STATE_SIZE,](torch.FloatTensor)
        '''
        self.sentence = []
        first_index = self.get_first_index(c)
        self.sentence.append(first_index)
        obs = self.encode(self.sentence)
        return obs

    def step(self, action, test=False):
        '''
        IN:
        action: single integer
        OUT:
        tuple (obs, 0, done, None) where
            obs: [STATE_SIZE,](torch.FloatTensor)
            done: single boolean value
        '''
        test=True
        self.sentence.append(action)
        if len(self.sentence) >= GEN_MAX_LEN:
            done = True
        else:
            if test and action == SEP_TOKEN_IDX:
                done = True
            else:
                done = False
        obs = self.encode(self.sentence)
        return obs, 0, done, None

    def get_first_index(self, code):
        if code == 0:
            r = random.randrange(self.neg_first_list[-1][1])
            for i in range(len(self.neg_first_list)):
                if(self.neg_first_list[i][1] <= r and r < self.neg_first_list[i+1][1]):
                    return self.neg_first_list[i][0]
        else:
            r = random.randrange(self.pos_first_list[-1][1])
            for i in range(len(self.pos_first_list)):
                if(self.pos_first_list[i][1] <= r and r < self.pos_first_list[i+1][1]):
                    return self.pos_first_list[i][0]

    def encode(self, sentence):
        original = copy(sentence)
        original.append(SEP_TOKEN_IDX)
        original.insert(0, CLS_TOKEN_IDX)
        converted = []
        for index in original:
            converted.append(self.to_bert_dict[index])
        converted = torch.LongTensor(converted).view(1, -1).to(DEVICE)
        if ENCODING_FLAG == 'LAST':
            with torch.no_grad():
                obs = self.bert_model(converted)[0][0][-1]
        elif ENCODING_FLAG == 'FIRST':
            with torch.no_grad():
                obs = self.bert_model(converted)[0][0][0]
        elif ENCODING_FLAG == 'LAST2':
            with torch.no_grad():
                obs = torch.mean(self.bert_model(converted)[-1][-2], axis=1).squeeze()
        else:
            print('error: invalid encoding flag')
            assert False
        return obs

    def id_to_string(self):
        converted = []
        for index in self.sentence:
            converted.append(self.to_bert_dict[index])
        tokens = []
        for index in converted:
            token = self.tokenizer._convert_id_to_token(index)
            tokens.append(token)
        return self.tokenizer.convert_tokens_to_string(tokens)
