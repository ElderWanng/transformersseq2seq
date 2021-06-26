python s2s.py --model_name_or_path \
facebook/bart-large \
--dataset_name \
xsum \
--output_dir \
~/tmp/tst-summarization \
--gpus 4 \
--num_beams 5 \
--batch_size \
8 \
--per_device_eval_batch_size \
32