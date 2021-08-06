import argparse
import os
import pathlib

import pandas as pd
from datasets import load_dataset, Dataset
from pandas import DataFrame
import random
import numpy
from spacy.lang.en import English
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration, DataCollatorForSeq2Seq
import torch
import re
# # assert tok.batch_decode(generated_ids, skip_special_tokens=True) == ['UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria']

per_device_eval_batch_size = 80


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name",
                    type=str,
                    default="xsum",
                    help="The name of the dataset to use (via the datasets library).",
                    )
parser.add_argument(
    "--dataset_config_name",
    type=str,
    default="3.0.0",
    help="The configuration name of the dataset to use (via the datasets library).",
)
parser.add_argument(
    "--save_path",
    type=str,
    default=pathlib.Path(os.getcwd(), "data").as_posix(),
    help="The configuration name of the dataset to use (via the datasets library).",
)

args = parser.parse_args()
save_path = args.save_path

os.makedirs(save_path, exist_ok=True)
raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, )
data_split = raw_datasets["train"]
df: DataFrame = data_split.to_pandas()

empty_indices = []
for line in df.iterrows():
    if len(line[1]["document"]) == 0:
        empty_indices.append(line[0])
df = df.drop(empty_indices)

# df = df.head(100)

summaries_list = df["summary"].to_list()


nlp = English()
mask_tokenizer = nlp.tokenizer


def mask(text):
    tokns = mask_tokenizer(text)
    tokns = [tok.text for tok in tokns]
    def helper():

        random_index = random.randint(0, len(tokns) - 1)

        span_len = numpy.random.poisson(lam=3)
        ##don't at end
        tokens_to_end = len(tokns) - random_index - 1

        span_len = min(span_len, tokens_to_end)

        if span_len > 0:
            del tokns[random_index:(random_index + span_len)]
            tokns.insert(random_index, "<placeholder>")
        elif span_len == 0:
            tokns.insert(random_index, "<placeholder>")

    n = random.randint(1,3)
    for _ in range(n):
        helper()

    counter = 0
    for i in range(len(tokns)):

        if tokns[i] == "<placeholder>":
            tokns[i] = f"<extra_id_{counter}>"
            counter += 1
    restext = " ".join(tokns)
    return restext

def _filter(decoded,text):
    num_to_replace = len(re.findall(r'<extra_id_\d+>', text))
    _ori = text[:]
    # print(t5_tokenizer.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
    # _txt = t5_tokenizer.decode(output[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    _txt = decoded.strip()
    spans = re.split(r'<extra_id_\d>', _txt)

    spans = [span.strip()  for span in spans if span!=""]

    # for i in range(min(num_to_replace,len(spans))):
    try:
        for i in range(min(num_to_replace,len(spans))):
            # print(spans[i])
            _ori = _ori.replace(f"<extra_id_{i}>", spans[i])
    except:
        print('===================')
        print(decoded)
        print(text)
        print('-------------------')

    return _ori


masked=[]
for s in tqdm(summaries_list):
    masked.append(mask(s))

df["masked"] = masked
empty_indices = []
for line in df.iterrows():
    if len(line[1]["document"]) == 0:
        empty_indices.append(line[0])
df = df.drop(empty_indices)

dataset = Dataset.from_pandas(df)

os.makedirs("prepared",exist_ok=True)
dataset.save_to_disk("prepared")