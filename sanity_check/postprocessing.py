import argparse
import os
import pathlib

import datasets
import nltk
import numpy as np
import pandas as pd
from datasets import load_metric, load_dataset
import json
import pandas
from nltk import ngrams


parser = argparse.ArgumentParser()
parser.add_argument("--result_file",type=str)
parser.add_argument("--multinli_path",type=str)
args = parser.parse_args()
multinli_path = args.multinli_path


datafiles = {}
datafiles["train"] = pathlib.Path(multinli_path, "multinli_1.0_train.jsonl").as_posix()
datafiles["validation"] = pathlib.Path(multinli_path, "multinli_1.0_dev_matched.jsonl").as_posix()
nli_dataset = datasets.load_dataset("json", data_files=datafiles)
nli_dataset = nli_dataset.filter(lambda example:example["gold_label"].startswith("ent"))
val = nli_dataset["validation"]
print(nli_dataset)
srcs = []
tgts = []
promptID = []
pairID = []
for example in val:
    srcs.append(example["sentence1"])
    tgts.append(example["sentence2"])
    promptID.append(example["promptID"])
    pairID.append(example["pairID"])


df = pd.DataFrame(data = {"srcs":srcs, "tgts":tgts, "pairID":pairID,"promptID":promptID})

resfile = args.result_file
os.makedirs(f"./{resfile.split()[0][:4]}",exist_ok=True)
save_path = f"./{resfile.split()[0][:4]}/data-dev.jsonl"
save_path2 = f"./{resfile.split()[0][:4]}/res.csv"
print(save_path)
with open(resfile,"r") as infile:
    res = json.load(infile)
print(len(res))
labels = [i[1] for i in res]
preds = [i[0] for i in res]
promptID = [i[2] for i in res]

df2 = pd.DataFrame(data = {"preds":preds, "labels":labels, "promptID":promptID})


joint_result = pd.concat([df, df2], join="inner")
print(df.sort_values('promptID')[["promptID","pairID"]])
print("------------------------")
print(df2.sort_values('promptID')[["promptID"]])
print("========================")
print(joint_result)