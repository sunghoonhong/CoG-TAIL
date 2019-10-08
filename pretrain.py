from Agent import Agent
from config import *
from util import *
EPOCHS = 10

if __name__ == '__main__':
    model, tokenizer = get_bert_model_and_tokenizer()
    agent = Agent(model)
    expert_chunk_generator = get_expert_chunk_generator()
    for epoch in range(EPOCHS):
        expert_chunk = next(expert_chunk_generator)
        loss = agent.pretrain(expert_chunk)
        print(loss)