set -e
NLI=/scratch/tw2112/datas/multinli_1.0
CACHE=/scratch/tw2112/TEMPCACHE
OUTDIR=/scratch/tw2112/exp_out/s2s

python preprocessing.py --multinli_path $NLI --out_dataset_dir $CACHE

python s2s_aux.py --model_name_or_path \
facebook/bart-base \
--output_dir \
$OUTDIR \
--data_on_disk_path \
$CACHE \
--gpus 4 \
--num_beams 5 \
--per_device_train_batch_size \
24