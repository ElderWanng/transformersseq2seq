set -e

MODEL=facebook/bart-large
TASK=ablation_only_label
NLI=$SCRATCH/datas/multinli_1.0
CACHE=$SCRATCH/TEMPCACHE
OUTDIR=$SCRATCH/exp_out/$TASK

python preprocessing_only_label.py --multinli_path $NLI --out_dataset_dir $CACHE

python Ablation.py --model_name_or_path \
$MODEL \
--output_dir \
$OUTDIR \
--data_on_disk_path \
$CACHE \
--gpus 4 \
--num_beams 6 \
--per_device_train_batch_size \
8 \
--per_device_eval_batch_size \
32 \
--label_smoothing \
0.1 \
--clip_norm \
0.1 \
--num_warmup_steps \
500 \
--learning_rate \
2e-5 \
--lr_scheduler_type \
cosine_with_restarts \
--weight_decay \
0.01 \
--task_name \
$TASK