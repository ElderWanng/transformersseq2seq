#MODEL_PATH=/scratch/zp2053/exp_out/ablation_only_task/s2s_with_aux-epoch=03-rouge1=44.83-v2.ckpt
MODEL_PATH=/scratch/tw2112/exp_out/ablation_only_label/s2s_with_aux-epoch=03-rouge1=44.83.ckpt
CACHE=$SCRATCH/TEMPCACHE/BART_MODEL

python extract_model.py --model_name_or_path $MODEL_PATH --output_dir $CACHE

