# -*- coding: utf-8 -*-


import os

import nltk
import numpy as np
import pandas as pd
from datasets import load_metric, load_dataset
import json
import pandas
from nltk import ngrams

rouge_metric = load_metric("rouge")
resfile = "sumNoneres.txt"
os.makedirs(f"./{resfile.split()[0][:4]}",exist_ok=True)
save_path = f"./{resfile.split()[0][:4]}/data-dev.jsonl"
save_path2 = f"./{resfile.split()[0][:4]}/res.csv"
print(save_path)
with open(resfile,"r") as infile:
    res = json.load(infile)
labels = [i[1] for i in res]
preds = [i[0] for i in res]
ids = [i[2] for i in res]
print(res[1])
xsum = load_dataset("xsum")
val = xsum["validation"]
df = pd.DataFrame(data = {"preds":preds, "labels":labels, "ids":ids})
srcs = []
tgts = []
ids = []
for example in val:
    srcs.append(example["document"])
    tgts.append(example["summary"])
    ids.append(int(example["id"]))
df2 = pd.DataFrame(data = {"srcs":srcs, "tgts":tgts, "ids":ids})

result = pd.concat([df, df2], join="inner")
r2 = pd.merge(df, df2, how='inner', on="ids")
r2 = r2.sort_values("ids")
r2.reindex()
with open(save_path, "w") as file:
    for e in r2.iterrows():
        # print(e[-1])
        e = e[-1]
        line = {}
        line["id"] = e["ids"]
        line["text"] = e["srcs"]
        line["claim"] = e["preds"]
        line["label"] = "CORRECT"
        file.write(json.dumps(line) + "\n")

r2.to_csv(save_path2, index=False,encoding="utf-8")
# def get_score(n):
#     res_score = []
#     # sixgrams = ngrams(sentence.split(), n)
#     for line in r2.iterrows():
#         # print(line)
#         line = line[1]
#         src = line.srcs
#         pred = line.preds
#         ngrams_pred = ngrams(nltk.tokenize.word_tokenize(pred), n)
#         ngrams_pred = list(ngrams_pred)
#         ngrams_src = ngrams(nltk.tokenize.word_tokenize(src), n)
#         ngrams_src = list(ngrams_src)
#         temp = [ngram in ngrams_src for ngram in ngrams_pred]
#         temp = np.array(temp)
#         res_score.append(temp.mean())
#         # print(temp.mean())
#     return np.array(res_score).mean()*100
# # print(np.array(res_score).mean())
# print(f"{get_score(1):.2f},{get_score(2):.2f}")