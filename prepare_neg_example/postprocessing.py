import argparse
import json
import os
import pathlib
import re

import pandas as pd
from datasets import load_dataset



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
        for i in range(num_to_replace):
            # print(spans[i])
            _ori = _ori.replace(f"<extra_id_{i}>", spans[i])
    except:
        print('===================')
        print(decoded)
        print(text)
        print('-------------------')

    return _ori
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoded",type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=pathlib.Path(os.getcwd(),"data").as_posix(),
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    args = parser.parse_args()
    save_path = args.save_path
    os.makedirs(save_path,exist_ok=True)

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, )

    data = raw_datasets[args.split]


    with open(args.decoded,'r') as input_file:
        decoded = json.load(input_file)

    df_decodede = pd.DataFrame(decoded, columns=['decoded', 'id'])
    print(df_decodede.columns)
    df_input = pd.read_csv(args.input_file)
    print(df_input.columns)
    newdf = df_decodede.merge(df_input, how="left", left_on=["id"], right_on=['id'])

    newdf = newdf.drop(columns="Unnamed: 0")
    newdf["nli"] = "con"
    print(newdf.columns)
    # newdf.head()
    #
    col_decoded = newdf['decoded'].tolist()
    col_summary = newdf['summary'].tolist()
    cleaned = []
    for i in range(len(col_decoded)):
        cleaned.append(_filter(col_decoded[i],col_summary[i]))
    newdf["cleaned"] =cleaned
    newdf.to_csv("tmp.csv")







