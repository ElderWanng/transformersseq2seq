from typing import Optional

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PLDataModule(pl.LightningDataModule):
    def __init__(self,
                 model_name_or_path,
                 train_file,
                 validation_file,
                 aux_file,
                 pad_to_max_length,
                 preprocessing_num_workers,
                 overwrite_cache,
                 max_source_length,
                 max_target_length,
                 train_batch_size,
                 val_batch_size,
                 dataloader_num_workers,
                 ignore_pad_token_for_loss):
        super(PLDataModule, self).__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.aux_file = aux_file
        self.model_name_or_path = model_name_or_path
        # self.line_by_line = line_by_line
        self.pad_to_max_length = pad_to_max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        # self.max_seq_length = max_seq_length
        # self.mlm_probability = mlm_probability
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss


    def setup(self, stage: Optional[str] = None) -> None:
        text_column = "src"
        summary_column = "tgt"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        data_files = {}
        if self.train_file is not None:
            data_files["train"] = self.train_file
        if self.validation_file is not None:
            data_files["validation"] = self.validation_file
        if self.aux_file is not None:
            data_files["auxiliary"] = self.aux_file
        extension = self.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
        padding = "max_length" if self.pad_to_max_length else False
        label_pad_token_id = -100 if self.ignore_pad_token_for_loss else self.tokenizer.pad_token_id

        processed_datasets = raw_datasets.map(
            self._preprocess_function,
            batched=True,
            remove_columns=[text_column,summary_column],
            load_from_cache_file=not self.overwrite_cache
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            # model=self.model,
            label_pad_token_id=label_pad_token_id,
            # pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )

    def _preprocess_function(self,examples):
        padding = "max_length" if self.pad_to_max_length else False
        label_pad_token_id = -100 if self.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        text_column = "src"
        summary_column = "tgt"

        inputs = examples[text_column]
        targets = examples[summary_column]
        # inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=padding,
                                      truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, padding=padding,
                                    truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

if __name__ == '__main__':
    # args = S2STransformer.parse_args()
    dm = PLDataModule(
        model_name_or_path= "facebook/bart-large-xsum" ,
        train_file="/Users/mac/Downloads/xsum.json/xsum.test.json",
        validation_file = "/Users/mac/Downloads/xsum.json/xsum.validation.json",
        aux_file=None,
        pad_to_max_length=False,
        preprocessing_num_workers = 1,
        overwrite_cache=False,
        max_source_length= 1024,
        max_target_length = 128,
        train_batch_size = 4,
        val_batch_size= 4,
        dataloader_num_workers=4,
        ignore_pad_token_for_loss=True
    )
    dm.setup()
    lder  = dm.train_dataloader()
    for i,b in enumerate(lder):
        if i==50:
            break
        print(b.keys())
    print(1)

