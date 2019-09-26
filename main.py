import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'#,1,2,3'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertModel, BertTokenizer
from sklearn.metrics import average_precision_score
#from isu_tool.ui import pgbar
import pickle, random, json, time, sys, math
from types import MethodType
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import subprocess
import argparse
from custom_models import *

data_dir = 'data/robust04'
model_dir = 'save/'

max_query_len = 20
max_doc_len = 800

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########## utils ##########
def time2str(time):
    time = int(time)
    h = time // 3600
    m = time % 3600 // 60
    s = time % 60
    ret = '%02dh%02dm%02ds' % (h, m, s)
    return ret
    
def load_json(path):
    print('Loading [Json|%s] ... ' % path, end='', flush=True)
    start_time = time.time()
    with open(path, 'r', encoding='utf-8') as fp:
        ret = json.loads(fp.read().strip())
    end_time = time.time()
    print('Finish! (%s)' % time2str(end_time - start_time), flush=True)
    return ret

def load_pickle(path):
    print('Loading [Pickle|%s] ... ' % path, end='', flush=True)
    start_time = time.time()
    with open(path, 'rb') as fp:
        ret = pickle.load(fp)
    end_time = time.time()
    print('Finish! (%s)' % time2str(end_time - start_time), flush=True)
    return ret

def get_metric(scores, labels):
    metric = {'map':0., 'mrr':0., 'p5':0., 'p10':0., 'r5':0., 'r10':0., 'r20':0.}
    merged = [(score, label) for score, label in zip(scores, labels)]
    merged = sorted(merged, key=lambda m: m[0], reverse=True)
    ranks = []
    for i, m in enumerate(merged):
        if m[1] > 0.5:
            ranks.append(i+1)
    metric['map'] = average_precision_score(labels, scores)
    metric['mrr'] = 1 / ranks[0]
    metric['p5']  = sum([m[1] for m in merged[:5]]) / 5
    metric['p10'] = sum([m[1] for m in merged[:10]]) / 10
    metric['r5']  = sum([m[1] for m in merged[:5]]) / sum(labels)
    metric['r10'] = sum([m[1] for m in merged[:10]]) / sum(labels)
    metric['r20'] = sum([m[1] for m in merged[:20]]) / sum(labels)

    return metric

def build_pair(data_train, data_agg, agg_ratio, oversample=1, shuffle=False):
    data_pair = []
    rel_data = []
    urel_data = []
    for data in data_train:
        dsize = data['dsize']
        rel_idx = []
        urel_idx = []
        for i in range(dsize):
            if data['rel_1d'][i] > 0.5:
                rel_idx.append(i)
                rel_data.append({'query': data['query'][i], 'docs': data['docs'][i]})
            else:
                urel_idx.append(i)
                urel_data.append({'query': data['query'][i], 'docs': data['docs'][i]})

        if len(rel_idx) * len(urel_idx) == 0:
            continue

        for rel_pick in rel_idx:
            for _ in range(oversample):
                urel_pick = random.choice(urel_idx)
                data_pair.append({
                    'rel': {'query': data['query'][rel_pick], 'docs': data['docs'][rel_pick]},# 'query_len': data['query_len'][rel_pick], 'docs_len': data['docs_len'][rel_pick]},
                    'urel': {'query': data['query'][urel_pick], 'docs': data['docs'][urel_pick]}# 'query_len': data['query_len'][urel_pick], 'docs_len': data['docs_len'][urel_pick]}
                })
    for data in data_agg:
        dsize = data['dsize']
        # Assume that all docs from data_agg are relevant
        for i in range(int(dsize * agg_ratio)):
            # for _ in range(oversample): # No oversampling for agg data
            urel_dict = random.choice(rel_data + urel_data)
            data_pair.append({
                'rel': {'query': data['query'][i], 'docs': data['docs'][i]},# 'query_len': data['query_len'][i], 'docs_len': data['docs_len'][i]},
                'urel': urel_dict
            })
                
    if shuffle:
        random.shuffle(data_pair)

    return data_pair

def build_batch(data_pair, batch, batch_size):
    rel_query, rel_docs = [], []
    urel_query, urel_docs = [], []

    for i in range(batch_size * batch, batch_size * (batch + 1)):
        rel_query.append(data_pair[i]['rel']['query'])
        rel_docs.append(data_pair[i]['rel']['docs'])
        urel_query.append(data_pair[i]['urel']['query'])
        urel_docs.append(data_pair[i]['urel']['docs'])

    rel_query = torch.stack(rel_query)
    rel_docs = torch.stack(rel_docs)
    urel_query = torch.stack(urel_query)
    urel_docs = torch.stack(urel_docs)
    rel_dict = {
        'query': rel_query,
        'docs': rel_docs
    }
    urel_dict = {
        'query': urel_query,
        'docs': urel_docs
    }
    return rel_dict, urel_dict

def trec_eval_custom(metrics, label_path, predict_path):
    eval_res = subprocess.Popen(['python', 'data/run_eval.py', label_path, predict_path], stdout=subprocess.PIPE, shell=False)
    (out, err) = eval_res.communicate()
    eval_res = out.decode("utf-8")
    results = {}

    for line in eval_res.split('\n'):
    
        splitted_line = line.split()
        try:
            first_element = splitted_line[0]
            for metric in metrics:
                if first_element == metric:
                    value = float(splitted_line[2])
                    results[metric] = value
        except:
            continue
    return results


criterion = nn.BCELoss()

def test(model, split, data_test, batch_size_test, skip_pred=False):
    model.eval()
    metric = {'map':0., 'mrr':0., 'p5':0., 'p10':0., 'r5':0., 'r10':0., 'r20':0.}
    res_dict = {'questions':[]}

    with torch.no_grad():
        for data in data_test:
        #for data in pgbar(data_test, pre='[ TEST %d ]' % split):
            if skip_pred:
                continue

            dsize = data['dsize']
            scores = []
            labels = []

            batch_len = dsize // batch_size_test
            if dsize % batch_size_test != 0:
                batch_len += 1

            for batch in range(batch_len):
                query_tok = data['query'][batch*batch_size_test:(batch+1)*batch_size_test]
                docs_tok = data['docs'][batch*batch_size_test:(batch+1)*batch_size_test]

                scores_t = model(query_tok.to(device), docs_tok.to(device)).view(-1)
                labels_t = data['rel_1d'][batch*batch_size_test:(batch+1)*batch_size_test].to(device)
                scores.append(scores_t)
                labels.append(labels_t)
            scores = torch.cat(scores).tolist()[:dsize]
            labels = torch.cat(labels).tolist()[:dsize]

            sorted_docs = sorted(zip(scores, data['docs_id']), reverse=True)
            res_dict['questions'].append({'id':data['query_id'], 'documents':[doc[1] for doc in sorted_docs]})

        with open('%s/split_%d/res_dict.json' % (data_dir, split), 'w') as fp:
            fp.write(json.dumps(res_dict))

        metric = trec_eval_custom(['map', 'P_10', 'P_20', 'ndcg_cut_10', 'ndcg_cut_20'], '%s/split_%d/rob04.test.s%d.json' % (data_dir, split, split), '%s/split_%d/res_dict.json' % (data_dir, split))

    return metric

def train(min_map,
        splits,
        epochs,
        Model,
        load_path,
        agg_path,
        verbose,
        batch_size,
        batch_size_test,
        model_name,
        agg_ratio,
        oversample):
    data_agg = load_pickle(agg_path)
    for split in splits:
        max_map = min_map
        # iter_cnt = 0

        model = Model()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        if load_path:
            model.load_state_dict(load_path)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

        data_train = load_pickle('%s/split_%d/prepro.train.pkl' % (data_dir, split))
        # data_dev = load_pickle('%s/split_%d/prepro.dev.pkl' % (data_dir, split))
        data_test = load_pickle('%s/split_%d/prepro.test.pkl' % (data_dir, split))
        for epoch in range(epochs):
            data_pair = build_pair(data_train, data_agg, agg_ratio, oversample, shuffle=True)
            losses = 0.
            batch_num = len(data_pair) // batch_size
            for batch in range(batch_num):
            #for batch in pgbar(range(len(data_pair) // batch_size), pre='[ SPLIT %d - TRAIN %d ]' % (split, epoch)):
                start = time.time()
                rels, urels = build_batch(data_pair, batch, batch_size)

                rel_scores = model(rels['query'].to(device), rels['docs'].to(device))
                urel_scores = model(urels['query'].to(device), urels['docs'].to(device))

                scores = torch.cat([rel_scores, urel_scores], dim=1)
                total_loss = torch.mean(1. - scores.softmax(dim=1)[:, 0])

                running_loss = total_loss.item()
                losses += running_loss
                ratio = batch / batch_num * 100.
                
                model.zero_grad()
                total_loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                
                end = time.time()
                if verbose:
                    running_time = end - start
                    total_time = running_time * batch_num
                    r_min = int(running_time // 60)
                    r_sec = int(running_time % 60)
                    t_min = int(total_time // 60)
                    t_sec = int(total_time % 60)
                    print('[S{:d}E{:d} {:.2f}%]'.format(split, epoch, ratio),
                        'L %.4f' % (running_loss),
                        '[{:2d}m{:2d}s/{:2d}m{:2d}s]'.format(r_min, r_sec, t_min, t_sec),
                        end='\r')
                    
                # iter_cnt += 1
                if (batch + 1) % (batch_num // 4) == 0:
                    print()
                    metric = test(model, split, data_test, batch_size_test)
                    for m in metric:
                        print('[ %3s ] %.6f' % (m.upper(), metric[m]))
                    if max_map < metric['map']:
                        max_map = max(max_map, metric['map'])
                        model.save('save/%s/%d_%.6f.pth' % (model_name, split, metric['map']))
                        print('##### SAVED (save/%s/%d_%.6f) #####' % (model_name, split, metric['map']))
                    model.train()
                    print('E%d: loss %.4f' % (epoch, losses / (batch_num //  4)))
                    losses = 0.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=RANKER.keys(), default='bert')
    parser.add_argument('--split', type=int, nargs='*', default=[1,2,3,4,5])
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--agg_ratio', type=float, default=0.2)
    parser.add_argument('--oversample', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--batch_size_test', type=int, default=50)
    parser.add_argument('--min_map', type=float, default=0.25)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--agg_path', type=str, default='data/wt.pkl')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()

    if args.name is None:
        args.name = args.model

    print(args)

    if not os.path.exists(model_dir + args.name):
        os.makedirs(model_dir + args.name)

    if args.test:
        model = RANKER[args.model]()
        model.to(device)
        if args.load_path:
            model.load_state_dict(args.load_path)
        for split in args.split:
            data_test = load_pickle('%s/split_%d/prepro.test.pkl' % (data_dir, split))
            test(
                model=model,
                split=split,
                data_test=data_test,
                batch_size_test=args.batch_size_test
            )
    else:    
        train(
            min_map=args.min_map,
            splits=args.split,
            epochs=args.epoch,
            Model=RANKER[args.model],
            load_path=args.load_path,
            agg_path=args.agg_path,
            verbose=args.verbose,
            batch_size=args.batch_size,
            batch_size_test=args.batch_size_test,
            model_name=args.name,
            agg_ratio=args.agg_ratio,
            oversample=args.oversample
        )



