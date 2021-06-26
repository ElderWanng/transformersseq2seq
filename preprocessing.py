import datasets
import argparse
import pathlib

from utils import summarization_name_mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',type=str, default="xsum")
    parser.add_argument('--multinli_path', type = str, required = True)
    parser.add_argument('--dataset_config_name',type=str)
    parser.add_argument('--out_dataset_dir', type=str,required=True)
    args = parser.parse_args()

    # main_prefix = "sum"
    main_dataset = datasets.load_dataset(args.dataset_name, args.dataset_config_name, )


    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    column_names = main_dataset["train"].column_names
    summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    def flatten_main(example):
        return {"source":example[text_column],"target":example[summary_column]}
    main_dataset = main_dataset.map(function=flatten_main,remove_columns=column_names)
    main_task_prefix = "sum"
    def flatten_main2(example):
        nli_prefix = "ent"
        prefix = f"[{main_task_prefix}] [{nli_prefix}] "
        return {"source":prefix + example['source'],"target":example['target']}

    column_names = main_dataset["train"].column_names
    main_dataset = main_dataset.map(function=flatten_main2, remove_columns=column_names)
    print(main_dataset)

    # D:\codeproject\NLP\data\multinli_1.0
    aux_task_prefix = "aux"
    datafiles = {}
    datafiles["train"] = pathlib.Path(args.multinli_path, "multinli_1.0_train.jsonl").as_posix()
    datafiles["validation"] = pathlib.Path(args.multinli_path, "multinli_1.0_dev_matched.jsonl").as_posix()
    nli_dataset = datasets.load_dataset("json",data_files=datafiles)
    def flatten_nli(example):
        return {"source":example['sentence1'],"target":example['sentence2'],"label":example["gold_label"]}
    nli_columns = nli_dataset["train"].column_names
    nli_dataset = nli_dataset.map(function=flatten_nli,remove_columns=nli_columns,num_proc=4)

    def flatten_nli2(example):
        nli_prefix = example["label"][:3]
        prefix = f"[{aux_task_prefix}] [{nli_prefix}] "
        return {"source":prefix + example['source'],"target":example['target']}

    nli_columns = nli_dataset["train"].column_names
    nli_dataset = nli_dataset.map(function=flatten_nli2, remove_columns=nli_columns, num_proc=1)

    print(nli_dataset)
    main_dataset.save_to_disk(dataset_dict_path=pathlib.Path(args.out_dataset_dir,"main_dataset").as_posix())
    nli_dataset.save_to_disk(dataset_dict_path=pathlib.Path(args.out_dataset_dir,"nli_dataset").as_posix())
    print(main_dataset["train"][4])
    print(nli_dataset["train"][4])


if __name__ == '__main__':
    main()