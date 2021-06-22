import itertools
from typing import Optional

import datasets
import nltk
import numpy as np
import pytorch_lightning as pl

import argparse
import logging
import os

# from accelerate import Accelerator
import torch.distributed
import wandb
from datasets import load_dataset, load_metric
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch.utils.data import DataLoader
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SchedulerType, DataCollatorForSeq2Seq,
)

# import utils
from data import PLDataModule
from utils import summarization_name_mapping, mykey

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
# datasets.arrow_dataset.Dataset._map()
class S2STransformer(pl.LightningModule):
    def __init__(self,hparams, *args, **kwargs):
        super(S2STransformer, self).__init__()
        # self.hparams = hparams
        self.save_hyperparameters()
        self.source_prefix = hparams.source_prefix
        self.model_name_or_path = hparams.model_name_or_path
        self.dataset_name = hparams.dataset_name
        self.dataset_config_name = hparams.dataset_config_name
        self.train_file = hparams.train_file
        self.validation_file = hparams.validation_file
        self.use_slow_tokenizer = hparams.use_slow_tokenizer
        self.overwrite_cache = hparams.overwrite_cache
        self.ignore_pad_token_for_loss = hparams.ignore_pad_token_for_loss
        self.pad_to_max_length = hparams.pad_to_max_length
        self.max_source_length = hparams.max_source_length
        self.max_target_length = hparams.max_target_length
        self.max_length = hparams.max_length
        self.output_dir = hparams.output_dir
        self.text_column = hparams.text_column
        self.summary_column = hparams.summary_column
        self.learning_rate = hparams.learning_rate
        self.per_device_train_batch_size = hparams.per_device_train_batch_size
        self.per_device_eval_batch_size = hparams.per_device_eval_batch_size
        self.dataloader_num_workers = hparams.dataloader_num_workers
        self.val_max_target_length = hparams.val_max_target_length
        self.num_beams = hparams.num_beams








        self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=not self.use_slow_tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_name_or_path),
            config=self.config,
        )

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.prefix = self.source_prefix if self.source_prefix is not None else ""





    def setup(self, stage: Optional[str] = None) -> None:

        # Sanity checks
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        if self.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(self.dataset_name, self.dataset_config_name, )
            dataset_columns = summarization_name_mapping.get(self.dataset_name, None)
        else:
            data_files = {}
            if self.train_file is not None:
                data_files["train"] = self.train_file
            if self.validation_file is not None:
                data_files["validation"] = self.validation_file
            # if self.aux_file is not None:
            if hasattr(self,"aux_file"):
                data_files["auxiliary"] = self.aux_file
            extension = self.train_file.split(".")[-1]
            raw_datasets = load_dataset(extension, data_files=data_files)

        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        dataset_columns = summarization_name_mapping.get(self.dataset_name, None)
        if self.text_column is None:
            text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            text_column = self.text_column
            if text_column not in column_names:
                raise ValueError(
                    f"--text_column' value '{self.text_column}' needs to be one of: {', '.join(column_names)}"
                )
        if self.summary_column is None:
            summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            summary_column = self.summary_column
            if summary_column not in column_names:
                raise ValueError(
                    f"--summary_column' value '{self.summary_column}' needs to be one of: {', '.join(column_names)}"
                )

        label_pad_token_id = -100 if self.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        pad_to_max_length = self.pad_to_max_length
        ignore_pad_token_for_loss = self.ignore_pad_token_for_loss
        tokenizer = self.tokenizer
        max_source_length = self.max_source_length
        max_target_length = self.max_target_length

        def preprocess_function(examples):
            padding = "max_length" if pad_to_max_length else False
            inputs = examples[text_column]
            targets = examples[summary_column]
            # inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding,
                                     truncation=True)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding,
                                   truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # for key,item in raw_datasets.items():
        #     raw_datasets[key] = raw_datasets[key][:200]
        processed_datasets = raw_datasets.map(
            function=preprocess_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=not self.overwrite_cache,
            num_proc=4
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            padding=True,
            # truncation=True
            # pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        # print(self.trainer)
        self.rouge_metric = load_metric('rouge',process_id=self.trainer.local_rank,num_process=self.trainer.world_size)
        # self.rouge_metric = load_metric('rouge')
        l = DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            collate_fn=self.data_collator,
            # num_workers=self.dataloader_num_workers,
            num_workers = 2
        )

        for i,batch in enumerate(l):
            print(i,batch)



    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )

    def forward(self, batch, batch_idx):

        return self.model(**batch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          self.learning_rate)
        # todo add add beta
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('train_loss', loss, on_epoch=True,on_step=True)
        return loss

    def _generate(self,batch):
        gen_kwargs = {
            "max_length": self.val_max_target_length,
            "num_beams": self.num_beams,
        }
        print("batch_size",batch["input_ids"].shape)
        generated_tokens = self.model.generate(input_ids = batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            **gen_kwargs)
        labels = batch["labels"]
        def pad_across_processes(data: torch.Tensor, dim=0, pad_index=0, pad_first=False):
            size = torch.tensor(data.shape, device=self.device)
            # size = generated_tokens.shape
            sizes = self.all_gather(size).cpu()
            if sizes.dim()>1:
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
        if self.ignore_pad_token_for_loss:
            labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)

        # generated_tokens = pad_across_processes(generated_tokens,dim=1, pad_index=self.tokenizer.pad_token_id)
        # generated_tokens = self.all_gather(generated_tokens)
        #
        # if self.ignore_pad_token_for_loss:
        #     labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # if not self.pad_to_max_length:
        #     # If we did not pad to max length, we need to pad the labels too
        #     labels = pad_across_processes(labels, dim=1, pad_index=self.tokenizer.pad_token_id)
        # labels = self.all_gather(labels)
        #
        # # print(generated_tokens.shape, f"from {self.trainer.local_rank}")
        # if self.trainer.world_size>1:
        #     generated_tokens = generated_tokens.reshape([generated_tokens.size(0) * generated_tokens.size(1), -1])
        #     labels = labels.reshape([labels.size(0) * labels.size(1), -1])

        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True,clean_up_tokenization_spaces=True)

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

            return preds, labels


        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        return decoded_preds, decoded_labels


    def validation_step(self,batch, batch_idx):
        # print(batch.keys(),self.trainer.local_rank)
        decoded_preds, decoded_labels = self._generate(batch)
        self.rouge_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def validation_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        res = None
        if self.trainer.world_size == 1:
            result = self.rouge_metric.compute(use_stemmer=True)
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            result = {k: round(v, 4) for k, v in result.items()}
            rouge1 = result["rouge1"]
            rouge2 = result["rouge2"]
            rougeL = result["rougeL"]
        elif self.trainer.world_size > 1:
            if self.local_rank == 0:
                result = self.rouge_metric.compute(use_stemmer=True)
                result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
                result = {k: round(v, 4) for k, v in result.items()}
                rouge1 = result["rouge1"]
                rouge2 = result["rouge2"]
                rougeL = result["rougeL"]
                res = torch.tensor([rouge1, rouge2, rougeL], device=self.device)
            else:
                res = torch.tensor([0.0, 0.0, 0.0], device=self.device)

            self.all_gather(res)

            res: torch.tensor = res.max(dim=0)
        res = res.item()
        rouge1 = res[0]
        rouge2 = res[1]
        rougeL = res[2]
        self.log("rouge1", rouge1, on_epoch=True, on_step=False)
        self.log("rouge2", rouge2, on_epoch=True, on_step=False)
        self.log("rougeL", rougeL, on_epoch=True, on_step=False)


    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        res = None
        if self.trainer.world_size==1:
            result = self.rouge_metric.compute(use_stemmer=True)
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            result = {k: round(v, 4) for k, v in result.items()}
            rouge1 = result["rouge1"]
            rouge2 = result["rouge2"]
            rougeL = result["rougeL"]

        elif self.trainer.world_size>1:
            if self.local_rank==0:
                result = self.rouge_metric.compute(use_stemmer=True)
                result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
                result = {k: round(v, 4) for k, v in result.items()}
                rouge1 = result["rouge1"]
                rouge2 = result["rouge2"]
                rougeL = result["rougeL"]
                res = torch.tensor([rouge1,rouge2,rougeL],device=self.device)
            else:
                res = torch.tensor([0.0, 0.0, 0.0], device=self.device)

            self.all_gather(res)

            res:torch.tensor = res.max(dim=0)
        res = res.item()
        rouge1 = res[0]
        rouge2 = res[1]
        rougeL = res[2]
        self.log("rouge1",rouge1,on_epoch=True,on_step=False)
        self.log("rouge2", rouge2, on_epoch=True, on_step=False)
        self.log("rougeL", rougeL, on_epoch=True, on_step=False)

    @staticmethod
    def add_model_specific_args(parser):
        # parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
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
            "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
        )
        parser.add_argument(
            "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
        )
        parser.add_argument(
            "--aux_file", type=str, default=None, help="A csv or a json file containing the aux data."
        )
        parser.add_argument(
            "--ignore_pad_token_for_loss",
            type=bool,
            default=True,
            help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
        )
        parser.add_argument(
            "--max_source_length",
            type=int,
            default=1024,
            help="The maximum total input sequence length after "
                 "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--source_prefix",
            type=str,
            default=None,
            help="A prefix to add before every source text " "(useful for T5 models).",
        )
        parser.add_argument(
            "--preprocessing_num_workers",
            type=int,
            default=None,
            help="The number of processes to use for the preprocessing.",
        )
        parser.add_argument(
            "--dataloader_num_workers",
            type=int,
            default=2,
            help="The number of dataloader_worker to use for the loader.",
        )
        parser.add_argument(
            "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
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
            default=None,
            help="Number of beams to use for evaluation. This argument will be "
                 "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
        )
        parser.add_argument(
            "--pad_to_max_length",
            action="store_true",
            help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
        )
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            help="Path to pretrained model or model identifier from huggingface.co/models.",
            # required=True,
            default="facebook/bart-large-xsum"
        )
        parser.add_argument(
            "--config_name",
            type=str,
            default=None,
            help="Pretrained config name or path if not the same as model_name",
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
            help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
        )
        parser.add_argument(
            "--per_device_train_batch_size",
            type=int,
            default=8,
            help="Batch size (per device) for the training dataloader.",
        )
        parser.add_argument(
            "--per_device_eval_batch_size",
            type=int,
            default=8,
            help="Batch size (per device) for the evaluation dataloader.",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=5e-5,
            help="Initial learning rate (after the potential warmup period) to use.",
        )
        parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
        parser.add_argument("--num_train_epochs", type=int, default=3,
                            help="Total number of training epochs to perform.")
        parser.add_argument(
            "--max_train_steps",
            type=int,
            default=None,
            help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument(
            "--lr_scheduler_type",
            type=SchedulerType,
            default="linear",
            help="The scheduler type to use.",
            choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        )
        parser.add_argument(
            "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
        )
        parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
        parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
        parser.add_argument(
            "--model_type",
            type=str,
            default=None,
            help="Model type to use if training from scratch.",
            choices=MODEL_TYPES,
        )


        return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--notification_email', type=str, default='will@email.com')

    # add model specific args
    parser = S2STransformer.add_model_specific_args(parser)
    argument = parser.parse_args()
    model = S2STransformer(argument)
    wandb.login(key=mykey)
    wandb_logger = WandbLogger(project='S2S', log_model='all')
    if torch.cuda.is_available():
        trainer = Trainer(gpus=argument.gpus,accelerator='ddp',logger=wandb_logger)
    else:
        trainer = Trainer(logger=wandb_logger)
    wandb_logger.watch(model)
    # trainer.fit(model)
    trainer.validate(model)


