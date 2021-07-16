CACHE=$SCRATCH/TEMPCACHE/BART_MODEL

python s2s_aux_ablation_decoding.py \
 --output_dir \
 $CACHE \
 --model_name \
 facebook/bart-large \
--dataset_name \
xsum \
--gpus $1 \
--num_beams 6 \
--task_prefix sum \
#--nli_prefix ent \

