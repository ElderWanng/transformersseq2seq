CACHE=$SCRATCH/TEMPCACHE/BART_MODEL

python s2s_aux_decoding.py \
 --output_dir \
 $CACHE \
 --model_name \
 facebook/bart-large \
--dataset_name \
xsum \
--gpus 4 \
--num_beams 6 \
--nli_prefix ent