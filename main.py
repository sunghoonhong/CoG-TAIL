import codecs
import torch
from Environment import Environment
from Agent import Agent
from config import *
from util import *

if __name__ == '__main__':
    bert_model, bert_tokenizer = get_bert_model_and_tokenizer()
    env = Environment(bert_model, bert_tokenizer)
    agent = Agent(bert_model)
    agent.pretrain_load()
    expert_chunk_generator = get_expert_chunk_generator()
    dist = torch.distributions.Categorical(probs=torch.full((CODE_SIZE,), fill_value=1/CODE_SIZE))
    for e in range(1000000):
        s = env.reset()
        c = dist.sample().numpy().item()
        d = False
        while not d:
            a, log_prob = agent.get_action(s, c)
            next_s, r, d, _ = env.step(a)
            time_to_update = agent.store(s, a, c, d, next_s, log_prob)
            if time_to_update:
                print('updating')
                expert_chunk = next(expert_chunk_generator)
                agent.update(expert_chunk)
                print('update complete')
            s = next_s
#        print(env.sentence)
        if e % TRAIN_REPORT_PERIOD == 0:
            if e == 0:
                f = codecs.open('train_generated_sentence.txt', 'w', "utf-8")
            else:
                f = codecs.open('train_generated_sentence.txt', 'a', 'utf-8')
            sentence = env.id_to_string()
            print(sentence)
            f.write('epoch: ' + str(e) + ' code: ' + str(c) + ' sentence: ' + sentence + '\n')
            f.close()


