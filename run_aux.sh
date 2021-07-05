set -e

MODEL=facebook/bart-large

NLI=$SCRATCH/datas/multinli_1.0
CACHE=$SCRATCH/TEMPCACHE
OUTDIR=$SCRATCH/exp_out/s2s

python preprocessing.py --multinli_path $NLI --out_dataset_dir $CACHE

python s2s_aux.py --model_name_or_path \
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
--lr_scheduler_type \
polynomial \
--weight_decay \
0.01