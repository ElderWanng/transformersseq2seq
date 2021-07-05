MODEL_PATH=/scratch/tw2112/exp_out/s2s/s2s_with_aux-epoch=04-rouge1=44.95.ckpt
CACHE=$SCRATCH/TEMPCACHE/BART_MODEL

python extract_model.py --model_name_or_path $MODEL_PATH --output_dir $CACHE