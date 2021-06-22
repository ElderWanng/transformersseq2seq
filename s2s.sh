python s2s_aux.py --model_name_or_path \
facebook/bart-large-xsum \
--dataset_name \
xsum \
--train_file \
/Users/mac/PycharmProjects/S2S/s2s-ft/data_prepocess/nli/nli.json \
--validation_file \
/Users/mac/PycharmProjects/S2S/s2s-ft/data_prepocess/nli/nli.json \
--aux_file \
/Users/mac/PycharmProjects/S2S/s2s-ft/data_prepocess/nli/nli.json \
--output_dir \
~/tmp/tst-summarization \
--gpus 1 \
--per_device_train_batch_size \
6 \
--dataloader_num_workers \
8
