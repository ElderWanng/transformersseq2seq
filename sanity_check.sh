CACHE=$SCRATCH/TEMPCACHE/BART_MODEL

python sanity_check.py \
 --output_dir \
 $CACHE \
 --model_name \
 facebook/bart-large \
--gpus 2 \
--num_beams 6 \
--nli_prefix ent \
--task_prefix aux \
--multinli_path \
$SCRATCH/datas/multinli_1.0 \
#--per_device_eval_batch_size 1