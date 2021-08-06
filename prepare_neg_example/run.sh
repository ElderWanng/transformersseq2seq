python t5_generate.py \
--model_name_or_path \
t5-base \
--dataset_name \
xsum \
--gpus 4 \
--num_beams 20 \
--per_device_eval_batch_size 64 \
--train_file \
data/xsum_train_data.csv \
--validation_file \
data/xsum_train_data.csv \
--save_name train.txt
