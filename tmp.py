import torch
import numpy as np


tmp = np.load('./IMDB_Dataset.npz')
states = tmp['states'][:512]
actions = tmp['actions'][:512]
codes = tmp['codes'][:512]
np.savez_compressed('./expert_data/tmp', states=states, actions=actions, codes=codes)
tmp = np.load('./expert_data/tmp.npz')
print(tmp['actions'].shape)
print(tmp['states'].shape)
print(tmp['codes'].shape)
