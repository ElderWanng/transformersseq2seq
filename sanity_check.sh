CACHE=$SCRATCH/TEMPCACHE/BART_MODEL

python sanity_check.py \
 --output_dir \
 $CACHE \
 --model_name \
 facebook/bart-large \
--gpus $1 \
--num_beams 6 \
--nli_prefix ent \
--task_prefix aux \
--multinli_path \
$SCRATCH/datas/multinli_1.0