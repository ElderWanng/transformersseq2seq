python s2s.py --model_name_or_path \
facebook/bart-base \
--dataset_name \
xsum \
--output_dir \
~/tmp/tst-summarization \
--gpus 4 \
--num_beams 5 \
--per_device_train_batch_size \
48