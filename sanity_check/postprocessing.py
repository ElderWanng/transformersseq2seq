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
val = nli_dataset["validation"]
print(nli_dataset)


df = val.to_pandas()
df["sentence2"] = df["sentence2"].apply(lambda x: x.strip())
resfile = args.result_file
os.makedirs(f"./{resfile.split()[0][:7]}",exist_ok=True)
save_path = f"./{resfile.split()[0][:7]}/data-dev.jsonl"
save_path2 = f"./{resfile.split()[0][:7]}/res.pkl"
print(save_path)
with open(resfile,"r") as infile:
    res = json.load(infile)
labels = [i[1].strip().replace("\n", " ") for i in res]
preds = [i[0].strip().replace("\n", " ") for i in res]
promptID = [str(i[2]) for i in res]
df2 = pd.DataFrame(data = {"preds":preds, "sentence2":labels, "ids":promptID})

df3 = pd.merge(df, df2,left_on = ["sentence2","promptID"],right_on = ["sentence2","ids"] )
with open(save_path, "w") as file:
    for e in df3.iterrows():
        e = e[-1]
        line = {}
        line["id"] = e["pairID"]
        line["text"] = e["sentence1"]
        line["claim"] = e["preds"]
        line["label"] = "CORRECT"
        file.write(json.dumps(line) + "\n")
df3.to_pickle(save_path2)