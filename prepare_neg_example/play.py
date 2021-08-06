import argparse
import os
import pathlib

import datasets
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

per_device_eval_batch_size = 100
num_beams = 12

T5_PATH = 't5-base' # "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU
t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_config = T5Config.from_pretrained(T5_PATH)
t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)


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
dataset = datasets.Dataset.load_from_disk("prepared")

print(dataset)

df: DataFrame = dataset.to_pandas()
# df = df.head(60)










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




max_source_length = 128
max_target_length = 128
def preprocess_function_ent(examples):
    padding = "max_length"
    inputs = examples["masked"]

    model_inputs = t5_tokenizer(inputs, max_length=max_source_length, padding=padding,
                                  truncation=True)

    with t5_tokenizer.as_target_tokenizer():
        labels = t5_tokenizer(inputs, max_length=max_target_length, padding=padding,
                                truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    # print(examples["id"])
    model_inputs["id"] = [int(num) for num in examples["id"]]
    return model_inputs

column_names = dataset.column_names

print("tokenizing dataset")
processed_dataset = dataset.map(
    function=preprocess_function_ent,
    batched=True,
    remove_columns=column_names,
)

data_collator = DataCollatorForSeq2Seq(
    t5_tokenizer,
    # model=self.model,
    # label_pad_token_id=label_pad_token_id,
    padding=True,
    # pad_to_multiple_of=8
)
dl = DataLoader(
            processed_dataset,
            batch_size=per_device_eval_batch_size,
            collate_fn=data_collator,)

result_df = pd.DataFrame(columns=['id','generated'])
print("generate")
for batch in tqdm(dl):
    input_ids = batch["input_ids"]
    input_ids = input_ids.to(DEVICE)


    outputs = t5_mlm.generate(input_ids=input_ids,
                              num_beams=num_beams, num_return_sequences=1)
    tokens =  t5_tokenizer.batch_decode(outputs[:,1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    ids = batch["id"].tolist()

    for i in range(len(ids)):
        result_df = result_df.append({"id":ids[i],"generated": tokens[i]},ignore_index=True)

print("generate done, postprocessing")
print(result_df)
df["id"]=df['id'].apply(int)
df2 = df.merge(result_df,how='inner',on='id')

print(df.columns)
print(df2)
df2["nli"] = "con"
col_decoded = df2['generated'].tolist()
col_summary = df2['masked'].tolist()
cleaned = []
for i in range(len(col_decoded)):
    cleaned.append(_filter(col_decoded[i], col_summary[i]))
df2["cleaned"] = cleaned
df2.to_csv("cleaned.csv")

