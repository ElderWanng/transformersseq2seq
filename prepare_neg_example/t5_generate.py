import argparse
import json
import os
import pathlib
from typing import Optional, Union, List

import nltk
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import load_dataset, load_metric
from pytorch_lightning import Trainer
# from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

from utils import summarization_name_mapping


class S2SDecode(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super(S2SDecode, self).__init__()
        self.save_hyperparameters()
        # print(self.hparams['hparams'])

        self.config = AutoConfig.from_pretrained(self.hparams['hparams'].model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams['hparams'].model_name_or_path, use_fast=not self.hparams['hparams'].use_slow_tokenizer)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hparams['hparams'].model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams['hparams'].model_name_or_path),
            config=self.config,
        )

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.res = []

    def setup(self, stage: Optional[str] = None) -> None:

        data_files = {}

        # train_data = pd.read_pickle(self.hparams['hparams'].train_file)
        data_files["train"] = self.hparams['hparams'].train_file

        # validation_data = pd.read_pickle(self.hparams['hparams'].train_file)
        data_files["validation"] = self.hparams['hparams'].validation_file
        # if self.aux_file is not None:
        extension = self.hparams['hparams'].train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

        if self.hparams['hparams'].val_max_target_length is None:
            self.hparams['hparams'].val_max_target_length = self.hparams['hparams'].max_target_length

        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        dataset_columns = summarization_name_mapping.get(self.hparams['hparams'].dataset_name, None)
        if self.hparams['hparams'].text_column is None:
            text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            text_column = self.hparams['hparams'].text_column
            if text_column not in column_names:
                raise ValueError(
                    f"--text_column' value '{self.hparams['hparams'].text_column}' needs to be one of: {', '.join(column_names)}"
                )
        if self.hparams['hparams'].summary_column is None:
            summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            summary_column = self.hparams['hparams'].summary_column
            if summary_column not in column_names:
                raise ValueError(
                    f"--summary_column' value '{self.hparams['hparams'].summary_column}' needs to be one of: {', '.join(column_names)}"
                )

        raw_datasets_valid = raw_datasets["validation"]

        label_pad_token_id = self.tokenizer.pad_token_id
        pad_to_max_length = self.hparams['hparams'].pad_to_max_length
        # ignore_pad_token_for_loss = self.ignore_pad_token_for_loss
        tokenizer = self.tokenizer
        max_source_length = self.hparams['hparams'].max_source_length
        max_target_length = self.hparams['hparams'].max_target_length



        def preprocess_function_ent(examples):
            padding = "max_length" if pad_to_max_length else False
            inputs = examples[summary_column]

            # targets = examples[summary_column]
            model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding,
                                     truncation=True)
            # Setup the tokenizer for targets
            # print(tokenizer.batch_decode(model_inputs["input_ids"]))
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(inputs, max_length=max_target_length, padding=padding,
                                   truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            # print(examples["id"])
            model_inputs["id"] = [int(num) for num in examples["id"]]
            return model_inputs

        processed_datasets_ori = raw_datasets_valid.map(
            function=preprocess_function_ent,
            batched=True,
            remove_columns=column_names,
            # num_proc=4
        )


        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            # label_pad_token_id=label_pad_token_id,
            padding=True,
            pad_to_multiple_of=8
        )

        self.ori_dataset = processed_datasets_ori
        self.data_collator = data_collator

        # print(len(self.ent_dataset))
        # self.rouge_metric = load_metric('rouge')

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.ori_dataset,
            batch_size=self.hparams['hparams'].per_device_eval_batch_size,
            collate_fn=self.data_collator,
        )

    def _generate(self, batch):
        gen_kwargs = {
            # "max_length": self.hparams['hparams'].val_max_target_length,
            "num_beams": self.hparams['hparams'].num_beams,
        }
        generated_tokens = self.model.generate(input_ids=batch["input_ids"],
                                               # attention_mask=batch["attention_mask"],
                                               **gen_kwargs)

        #drop first pad
        generated_tokens = generated_tokens[:,1:]
        # labels = batch["labels"]
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
        # labels = pad_across_processes(labels, dim=1)
        # labels = self.all_gather(labels)
        ids = self.all_gather(ids)
        generated_tokens = generated_tokens.view(-1, generated_tokens.shape[-1])
        # labels = labels.view(-1, labels.shape[-1])
        ids = ids.view(-1)
        # print(generated_tokens.shape, labels.shape, ids.shape, f"from rank {self.local_rank}")
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True,
        #                                              clean_up_tokenization_spaces=True)

        def postprocess_text(preds):
            preds = [pred.strip() for pred in preds]
            # labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

            return preds

        decoded_preds = postprocess_text(decoded_preds)

        self.res = self.res + list(zip(decoded_preds, ids.cpu().numpy().tolist()))
        # return decoded_preds, decoded_labels

    def validation_step(self, batch, batch_idx):
        self._generate(batch)
        # self.rouge_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def validation_epoch_end(self, outputs) -> None:

        if self.local_rank == 0:
            # labels = [i[1] for i in self.res]
            # preds = [i[0] for i in self.res]
            # self.rouge_metric.add_batch(predictions=preds, references=labels)
            # result = self.rouge_metric.compute(use_stemmer=True)
            # if type(result) is dict:
            #     result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            #     result = {k: round(v, 4) for k, v in result.items()}
            #     result = [score for name, score in result.items()]
            #     print(result)
                # print(self.res)
            with open(self.hparams['hparams'].save_name, "w") as outputfile:
                json.dump(self.res, outputfile)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            help="Path to pretrained model or model identifier from huggingface.co/models.",
            # required=True,
            default="facebook/bart-large-xsum"
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
        parser.add_argument(
            "--train_file",
            type=str,
            default="",
            # help="Batch size (per device) for the evaluation dataloader.",
        )
        parser.add_argument(
            "--validation_file",
            type=str,
            default="",
            # help="Batch size (per device) for the evaluation dataloader.",
        )
        parser.add_argument(
            "--save_name",
            type=str,
            required=True,
            # help="Batch size (per device) for the evaluation dataloader.",
        )




        return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--gpus', type=int, default=1)
    parser = S2SDecode.add_model_specific_args(parser)
    argument = parser.parse_args()
    model = S2SDecode(argument)
    if torch.cuda.is_available() and argument.gpus > 1:
        trainer = Trainer(
            gpus=argument.gpus,
            accelerator='ddp',
            # # logger=wandb_logger,
            # gradient_clip_val = argument.clip_norm,
            # precision=16,
            # # callbacks=[checkpoint_callback,lr_monitor],
            # val_check_interval=0.25
        )
    else:
        trainer = Trainer(
            # logger=wandb_logger,
            # callbacks=[checkpoint_callback,lr_monitor],
            # val_check_interval=0.25
        )

    trainer.validate(model)