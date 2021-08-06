python decode_s2s.py \
--model_name_or_path \
facebook/bart-large-xsum \
--dataset_name \
xsum \
--gpus 4 \
--num_beams 5 \
--per_device_eval_batch_size 4