python s2s.py --model_name_or_path \
facebook/bart-large \
--dataset_name \
xsum \
--output_dir \
$SCRATCH/exp_out \
--gpus 4 \
--num_beams 5 \
--batch_size \
8 \
--per_device_eval_batch_size \
32 \
--label_smoothing \
0.1 \
--clip_norm \
0.1 \
--num_warmup_steps \
1000 \
--lr_scheduler_type \
polynomial \
--weight_decay \
0.01