import nltk
import numpy as np
import pandas as pd
from datasets import load_metric, load_dataset
import json
import pandas
from nltk import ngrams

rouge_metric = load_metric("rouge")
with open("neures.txt","r") as infile:
    res = json.load(infile)
labels = [i[1] for i in res]
preds = [i[0] for i in res]
ids = [i[2] for i in res]
# rouge_metric.add_batch(predictions=preds, references=labels)
# result = rouge_metric.compute(use_stemmer=True)
# result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
# result = {k: round(v, 4) for k, v in result.items()}
# result = [score for name, score in result.items()]
# print(result)
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
with open(f"data-dev.jsonl", "w") as file:
    for e in r2.iterrows():
        # print(e[-1])
        e = e[-1]
        line = {}
        line["id"] = e["ids"]
        line["text"] = e["srcs"]
        line["claim"] = e["preds"]
        line["label"] = "CORRECT"
        file.write(json.dumps(line) + "\n")

n = 1


def get_score(n):
    res_score = []
    # sixgrams = ngrams(sentence.split(), n)
    for line in r2.iterrows():
        # print(line)
        line = line[1]
        src = line.srcs
        pred = line.preds
        ngrams_pred = ngrams(nltk.tokenize.word_tokenize(pred), n)
        ngrams_pred = list(ngrams_pred)
        ngrams_src = ngrams(nltk.tokenize.word_tokenize(src), n)
        ngrams_src = list(ngrams_src)
        temp = [ngram in ngrams_src for ngram in ngrams_pred]
        temp = np.array(temp)
        res_score.append(temp.mean())
        # print(temp.mean())
    return np.array(res_score).mean()*100
# print(np.array(res_score).mean())
print(f"{get_score(1):.2f},{get_score(2):.2f}")