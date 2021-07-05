import itertools
import pathlib
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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

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
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
)

# import utils
from data import PLDataModule
from multitask_dataset import MultiTaskDataset
from utils import summarization_name_mapping, mykey, loss_label_smoothing

import sys
# its win32, maybe there is win64 too?
if sys.platform.startswith('win'):
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "Gloo"
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
        self.model_name_or_path = hparams.model_name_or_path
        self.source_prefix = hparams.source_prefix

        self.data_on_disk_path = hparams.data_on_disk_path
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
        self.label_smoothing = hparams.label_smoothing
        self.learning_rate = hparams.learning_rate
        self.num_warmup_steps = hparams.num_warmup_steps
        self.per_device_train_batch_size = hparams.per_device_train_batch_size
        self.lr_scheduler_type = hparams.lr_scheduler_type
        self.per_device_eval_batch_size = hparams.per_device_eval_batch_size
        # self.dataloader_num_workers = hparams.dataloader_num_workers
        self.val_max_target_length = hparams.val_max_target_length
        self.num_beams = hparams.num_beams

        self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=not self.use_slow_tokenizer)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[sum]", "[aux]", '[ent]', "[con]",'[neu]']})
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_name_or_path),
            config=self.config,
        )

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.prefix = self.source_prefix if self.source_prefix is not None else ""
        self.label_pad_token_id = -100 if self.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        self.weight_decay = hparams.weight_decay





    def setup(self, stage: Optional[str] = None) -> None:



        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

        data_on_disk_path = self.data_on_disk_path
        main_dataset_path = pathlib.Path(data_on_disk_path,"main_dataset").as_posix()
        nli_dataset_path = pathlib.Path(data_on_disk_path, "nli_dataset").as_posix()
        main_dataset = datasets.load_from_disk(main_dataset_path)
        nli_dataset = datasets.load_from_disk(nli_dataset_path)

        print(main_dataset["train"][4])
        print(nli_dataset["train"][4])


        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

        # First we tokenize all the texts.
        main_column_names = main_dataset["train"].column_names
        text_column = main_column_names[0]
        summary_column = main_column_names[1]
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


        nli_column_names = nli_dataset["train"].column_names
        processed_main_datasets = main_dataset.map(
            function=preprocess_function,
            batched=True,
            remove_columns=main_column_names,
            load_from_cache_file=not self.overwrite_cache,
            num_proc=4
        )
        processed_nli_datasets = nli_dataset.map(
            function=preprocess_function,
            batched=True,
            remove_columns=nli_column_names,
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
        train_main_dataset = processed_main_datasets["train"]
        train_nli_dataset = processed_nli_datasets["train"]

        eval_dataset = processed_main_datasets["validation"]


        mtds = MultiTaskDataset([train_main_dataset,train_nli_dataset])
        print(mtds.rs)


        self.train_dataset = mtds
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        print(len(self.train_dataset))

        self.rouge_metric = load_metric('rouge',process_id=self.trainer.local_rank,num_process=self.trainer.world_size,experiment_id="My_experiment_10",cache_dir="./.cache")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.per_device_eval_batch_size,
            collate_fn=self.data_collator,
        )

    def forward(self, batch, batch_idx):

        return self.model(**batch)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          self.learning_rate)
        # todo add beta
        lr_scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=95000
        )
        lr_dict = {
            # REQUIRED: The scheduler instance
            'scheduler': lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            'interval': 'step',
            # # How many epochs/steps should pass between calls to
            # # `scheduler.step()`. 1 corresponds to updating the learning
            # # rate after every epoch/step.
            # 'frequency': 1,
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_dict
        }

    def training_step(self, batch, batch_idx):
        result = self.model(**batch)
        if self.label_smoothing==0:
            loss = result.loss
        else:
            labels = batch["labels"]
            loss = loss_label_smoothing(result, labels,self.label_pad_token_id, self.label_smoothing)
            self.log('original_train_loss', result.loss, on_epoch=True, on_step=True)
        self.log('train_loss', loss, on_epoch=True,on_step=True)
        return loss

    def _generate(self,batch):
        gen_kwargs = {
            "max_length": self.val_max_target_length,
            "num_beams": self.num_beams,
        }
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
        decoded_preds, decoded_labels = self._generate(batch)
        self.rouge_metric.add_batch(predictions=decoded_preds, references=decoded_labels)


    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        result = self.rouge_metric.compute(use_stemmer=True)
        if type(result) is dict:
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            result = {k: round(v, 4) for k, v in result.items()}
            result = [score for name,score in result.items()]
        else:
            result = [0, 0, 0, 0]

        result = torch.tensor(result, device=self.device,dtype=torch.float32)


        result = self.all_gather(result)
        # print(f"result after gather: rank is {self.local_rank},{result}")
        result = result.view(-1,4)
        # print(result, f"result after view: rank is {self.local_rank}")
        result = torch.max(result,dim=0,keepdim=False)[0].cpu().numpy()
        if self.local_rank == 0:
            print(result, f"rank is {self.local_rank}")
        self.log("rouge1", result[0], on_epoch=True, on_step=False)
        self.log("rouge2", result[1], on_epoch=True, on_step=False)
        self.log("rougeL", result[2], on_epoch=True, on_step=False)
        self.log("rougeSum", result[3], on_epoch=True, on_step=False)


    @staticmethod
    def add_model_specific_args(parser):
        # # parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
        # parser.add_argument(
        #     "--dataset_name",
        #     type=str,
        #     default=None,
        #     help="The name of the dataset to use (via the datasets library).",
        # )
        # parser.add_argument(
        #     "--dataset_config_name",
        #     type=str,
        #     default=None,
        #     help="The configuration name of the dataset to use (via the datasets library).",
        # )
        # parser.add_argument(
        #     "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
        # )
        # parser.add_argument(
        #     "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
        # )
        # parser.add_argument(
        #     "--aux_file", type=str, default=None, help="A csv or a json file containing the aux data."
        # )
        parser.add_argument(
            "--data_on_disk_path",
            type=str,
            default=None,
            help="the path to store main and nli datasets path",
            required=True
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
            "--label_smoothing",
            type=float,
            default=0,
            help="if 0 then use default loss from model.forward, otherwise compute loss twice",
        )
        parser.add_argument(
            "--clip_norm",
            type=float,
            default=0.0,
            help="clip gradient respect to norm of weight in case of gradient explosion",
        )
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
            type=str,
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
    # parser.add_argument('--notification_email', type=str, default='will@email.com')

    # add model specific args
    parser = S2STransformer.add_model_specific_args(parser)
    argument = parser.parse_args()
    model = S2STransformer(argument)
    wandb.login(key=mykey)
    wandb_logger = WandbLogger(project='S2S_WITH_AUX', log_model='all')
    checkpoint_callback = ModelCheckpoint(
        monitor='rouge1',
        verbose=True,
        dirpath= argument.output_dir,
        filename='s2s_with_aux-{epoch:02d}-{rouge1:.2f}',
        save_top_k=5,
        mode='max',
        every_n_val_epochs=1
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    if torch.cuda.is_available():
        trainer = Trainer(gpus=argument.gpus,
                          accelerator='ddp',
                          logger=wandb_logger,
                          gradient_clip_val = argument.clip_norm,
                          precision=16,
                          callbacks=[checkpoint_callback,lr_monitor],
                          val_check_interval=0.25
                          )
    else:
        trainer = Trainer(logger=wandb_logger,
                          callbacks=[checkpoint_callback,lr_monitor],
                          val_check_interval=0.25
                          )
    wandb_logger.watch(model)
    trainer.fit(model)
    # trainer.validate(model


