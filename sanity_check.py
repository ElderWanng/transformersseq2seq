import argparse
import json
import os
import pathlib
from typing import Optional, Union, List

import datasets
import nltk
import pytorch_lightning as pl
import torch
from datasets import load_dataset, load_metric
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

from utils import summarization_name_mapping


class S2SAuxDecode_SANITY(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super(S2SAuxDecode_SANITY, self).__init__()
        self.save_hyperparameters()
        # print(self.hparams['hparams'])
        model_output_name = pathlib.Path(self.hparams['hparams'].output_dir, "model.bin").as_posix()
        config_output_name = pathlib.Path(self.hparams['hparams'].output_dir, "config").as_posix()
        tokenizer_output_name = pathlib.Path(self.hparams['hparams'].output_dir, "tokenizer").as_posix()

        self.config = AutoConfig.from_pretrained(config_output_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams['hparams'].model_name,
                                                       use_fast=not self.hparams['hparams'].use_slow_tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_output_name,
            from_tf=bool(".ckpt" in model_output_name),
            config=self.config,
        )

        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[sum]", "[aux]", '[ent]', "[con]", '[neu]']})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.res = []

    def setup(self, stage: Optional[str] = None) -> None:

        # # Sanity checks
        # if self.hparams['hparams'].dataset_name is None and self.hparams['hparams'].train_file is None and self.hparams['hparams'].validation_file is None:
        #     raise ValueError("Need either a dataset name or a training/validation file.")
        # else:
        #     if self.hparams['hparams'].train_file is not None:
        #         extension = self.train_file.split(".")[-1]
        #         assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        #     if self.validation_file is not None:
        #         extension = self.validation_file.split(".")[-1]
        #         assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

        # if self.output_dir is not None:
        #     os.makedirs(self.output_dir, exist_ok=True)
        # if self.hparams['hparams'].dataset_name is not None:
        #     # Downloading and loading a dataset from the hub.
        #     raw_datasets = load_dataset(self.hparams['hparams'].dataset_name,
        #                                 self.hparams['hparams'].dataset_config_name, )
        #     dataset_columns = summarization_name_mapping.get(self.hparams['hparams'].dataset_name, None)
        # else:
        #     data_files = {}
        #     if self.hparams['hparams'].train_file is not None:
        #         data_files["train"] = self.hparams['hparams'].train_file
        #     if self.hparams['hparams'].validation_file is not None:
        #         data_files["validation"] = self.hparams['hparams'].validation_file
        #     # if self.aux_file is not None:
        #     extension = self.hparams['hparams'].train_file.split(".")[-1]
        #     raw_datasets = load_dataset(extension, data_files=data_files)

        label2id = {'contradiction': 0, 'entailment': 2, 'neutral': 1}
        id2label = {v:k for k,v in label2id.items()}
        datafiles = {}
        datafiles["train"] = pathlib.Path(self.hparams['hparams'].multinli_path, "multinli_1.0_train.jsonl").as_posix()
        datafiles["validation"] = pathlib.Path(self.hparams['hparams'].multinli_path, "multinli_1.0_dev_matched.jsonl").as_posix()
        raw_datasets = datasets.load_dataset("json", data_files=datafiles)
        print(raw_datasets)
        def flatten_nli(example):
            return {"source": example['sentence1'], "target": example['sentence2'], "gold_label": example["gold_label"],"id":example["promptID"]}

        nli_columns = raw_datasets["train"].column_names
        raw_datasets = raw_datasets.map(function=flatten_nli, remove_columns=nli_columns, num_proc=4)
        raw_datasets = raw_datasets.filter(lambda example:example["gold_label"].startswith("ent"))
        print(raw_datasets)



        if self.hparams['hparams'].nli_prefix is not None:
            nli_prefix = f"[{self.hparams['hparams'].nli_prefix}]"
        else:
            nli_prefix = None

        if self.hparams['hparams'].task_prefix is not None:
            task_prefix = f"[{self.hparams['hparams'].task_prefix}]"
        else:
            task_prefix = None
        prefixs = [task_prefix, nli_prefix]
        prefixs = [item for item in prefixs if item is not None]
        prefixs = " ".join(prefixs) + " "
        print(prefixs)


        def flatten_nli2(example):
            return {"source": prefixs + example['source'], "target": example['target'],"id":example["id"],"gold_label":label2id[example["gold_label"]]}

        nli_columns = raw_datasets["train"].column_names
        raw_datasets = raw_datasets.map(function=flatten_nli2, remove_columns=nli_columns, num_proc=1)


        raw_datasets_valid = raw_datasets["validation"]
        label_pad_token_id = self.tokenizer.pad_token_id
        pad_to_max_length = self.hparams['hparams'].pad_to_max_length
        # ignore_pad_token_for_loss = self.ignore_pad_token_for_loss
        tokenizer = self.tokenizer
        max_source_length = self.hparams['hparams'].max_source_length
        max_target_length = self.hparams['hparams'].max_target_length


        text_column = "source"
        summary_column = "target"
        def preprocess_function_ent(examples):
            padding = "max_length" if pad_to_max_length else False
            inputs = examples[text_column]
            targets = examples[summary_column]
            model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding,
                                     truncation=True)
            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding,
                                   truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            # print(examples["id"])
            model_inputs["id"] = [int(num) for num in examples["id"]]
            model_inputs["gold_label"] = [int(num) for num in examples["gold_label"]]
            return model_inputs

        column_names = raw_datasets_valid.column_names
        processed_datasets_ent = raw_datasets_valid.map(
            function=preprocess_function_ent,
            batched=True,
            remove_columns=column_names,
            num_proc=4
        )
        for i in range(20):
            ex = processed_datasets_ent[i]
            # print(ex)
            print(tokenizer.decode(ex["input_ids"]))
            print(tokenizer.decode(ex["labels"]))
            print("------------------------------------------------")

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            padding=True,
            pad_to_multiple_of=8
        )

        self.ent_dataset = processed_datasets_ent
        self.data_collator = data_collator

        # print(len(self.ent_dataset))
        self.rouge_metric = load_metric('rouge')

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.ent_dataset,
            batch_size=self.hparams['hparams'].per_device_eval_batch_size,
            collate_fn=self.data_collator,
        )

    def _generate(self, batch):
        gen_kwargs = {
            "max_length": self.hparams['hparams'].val_max_target_length,
            "num_beams": self.hparams['hparams'].num_beams,
        }
        generated_tokens = self.model.generate(input_ids=batch["input_ids"],
                                               attention_mask=batch["attention_mask"],
                                               **gen_kwargs)

        labels = batch["labels"]
        ids = batch["id"]

        # print(generated_tokens.shape, labels.shape, ids.shape)
        def pad_across_processes(data: torch.Tensor, dim=0, pad_index=0, pad_first=False):
            size = torch.tensor(data.shape, device=self.device)
            # size = generated_tokens.shape
            sizes = self.all_gather(size).cpu()
            if sizes.dim() > 1:
                max_size = max(s[dim].item() for s in sizes)
            else:
                max_size = sizes[dim].item()
            if max_size == data.shape[dim]:
                return data
            old_size = data.shape
            new_size = list(old_size)
            new_size[dim] = max_size
            new_tensor = data.new_zeros(tuple(new_size)) + pad_index
            if pad_first:
                indices = tuple(
                    slice(max_size - old_size[dim], max_size) if i == dim else slice(None) for i in range(len(new_size))
                )
            else:
                indices = tuple(slice(0, old_size[dim]) if i == dim else slice(None) for i in range(len(new_size)))
            new_tensor[indices] = data
            return new_tensor

        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]

        generated_tokens = pad_across_processes(generated_tokens, dim=1)
        generated_tokens = self.all_gather(generated_tokens)
        labels = pad_across_processes(labels, dim=1)
        labels = self.all_gather(labels)
        ids = self.all_gather(ids)
        generated_tokens = generated_tokens.view(-1, generated_tokens.shape[-1])
        labels = labels.view(-1, labels.shape[-1])
        ids = ids.view(-1)
        # print(generated_tokens.shape, labels.shape, ids.shape, f"from rank {self.local_rank}")
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

            return preds, labels

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        self.res = self.res + list(zip(decoded_preds, decoded_labels, ids.cpu().numpy().tolist()))
        # return decoded_preds, decoded_labels

    def validation_step(self, batch, batch_idx):
        self._generate(batch)
        # self.rouge_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        # print(self.res)
        # with open("res.txt","w") as outputfile:
        #     json.dump(self.res,outputfile)
        if self.local_rank == 0:
            labels = [i[1] for i in self.res]
            preds = [i[0] for i in self.res]
            self.rouge_metric.add_batch(predictions=preds, references=labels)
            result = self.rouge_metric.compute(use_stemmer=True)
            if type(result) is dict:
                result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
                result = {k: round(v, 4) for k, v in result.items()}
                result = [score for name, score in result.items()]
                print(result)
                # print(self.res)
                with open(f"{self.hparams['hparams'].nli_prefix}_sanity_check_res.txt", "w") as outputfile:
                    json.dump(self.res, outputfile)

    @staticmethod
    def add_model_specific_args(parser):

        parser.add_argument(
            "--model_name",
            type=str,
            default=None,
            # help="Pretrained tokenizer name or path if not the same as model_name",
        )

        parser.add_argument(
            "--dataset_name",
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
            "--max_source_length",
            type=int,
            default=1024,
            help="The maximum total input sequence length after "
                 "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--preprocessing_num_workers",
            type=int,
            default=None,
            help="The number of processes to use for the preprocessing.",
        )

        parser.add_argument(
            "--max_target_length",
            type=int,
            default=128,
            help="The maximum total sequence length for target text after "
                 "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
                 "during ``evaluate`` and ``predict``.",
        )
        parser.add_argument(
            "--val_max_target_length",
            type=int,
            default=None,
            help="The maximum total sequence length for validation "
                 "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
                 "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
                 "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
        )
        parser.add_argument(
            "--max_length",
            type=int,
            default=128,
            help=(
                "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
            ),
        )
        parser.add_argument(
            "--num_beams",
            type=int,
            default=5,
            help="Number of beams to use for evaluation. This argument will be "
                 "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
        )

        parser.add_argument(
            "--pad_to_max_length",
            action="store_true",
            help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
        )

        parser.add_argument(
            "--tokenizer_name",
            type=str,
            default=None,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )

        parser.add_argument(
            "--text_column",
            type=str,
            default=None,
            help="The name of the column in the datasets containing the full texts (for summarization).",
        )
        parser.add_argument(
            "--summary_column",
            type=str,
            default=None,
            help="The name of the column in the datasets containing the summaries (for summarization).",
        )

        parser.add_argument(
            "--use_slow_tokenizer",
            action="store_false",
            help="If passed, will use a slow tokenizer (not backed by the ???? Tokenizers library).",
        )

        parser.add_argument(
            "--per_device_eval_batch_size",
            type=int,
            default=32,
            help="Batch size (per device) for the evaluation dataloader.",
        )
        parser.add_argument("--output_dir", type=str, default=None, help="Where to load the model.")
        parser.add_argument("--nli_prefix", type=str, default=None, help="nli prefix to dataloader.")
        parser.add_argument("--task_prefix", type=str, default=None, help="task prefix to dataloader.")
        # nli_prefix

        parser.add_argument('--multinli_path', type=str, required=True)


        return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--gpus', type=int, default=1)
    parser = S2SAuxDecode_SANITY.add_model_specific_args(parser)
    argument = parser.parse_args()
    model = S2SAuxDecode_SANITY(argument)
    if torch.cuda.is_available() and argument.gpus > 1:
        trainer = Trainer(
            gpus=argument.gpus,
            accelerator='ddp',
        )
    else:
        trainer = Trainer(
            # logger=wandb_logger,
            # callbacks=[checkpoint_callback,lr_monitor],
            # val_check_interval=0.25
        )

    trainer.validate(model)