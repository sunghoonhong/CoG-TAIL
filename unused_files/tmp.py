import pickle
import gzip
import codecs


with gzip.open('Top5000_BtoA.pickle') as f:
    from_bert_dict = pickle.load(f)

with gzip.open('Top5000_AtoB.pickle') as f:
    to_bert_dict = pickle.load(f)

print(from_bert_dict[102], from_bert_dict[1029], from_bert_dict[999], from_bert_dict[1012], from_bert_dict[1008], from_bert_dict[2184])
print(to_bert_dict[0], to_bert_dict[1])
print(len(to_bert_dict), len(from_bert_dict))