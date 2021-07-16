import argparse

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from s2s_aux import S2STransformer
from utils import mykey

parser = argparse.ArgumentParser()

# add PROGRAM level args
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--task_name', type=str, default='Ablation')

# add model specific args
parser = S2STransformer.add_model_specific_args(parser)
argument = parser.parse_args()
model = S2STransformer(argument)
wandb.login(key=mykey)
wandb_logger = WandbLogger(project=argument.task_name, log_model='all')
checkpoint_callback = ModelCheckpoint(
    monitor='rouge1',
    verbose=True,
    dirpath=argument.output_dir,
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
                      gradient_clip_val=argument.clip_norm,
                      precision=16,
                      callbacks=[checkpoint_callback, lr_monitor],
                      val_check_interval=0.25
                      )
else:
    trainer = Trainer(logger=wandb_logger,
                      callbacks=[checkpoint_callback, lr_monitor],
                      val_check_interval=0.25
                      )
wandb_logger.watch(model)
trainer.fit(model)