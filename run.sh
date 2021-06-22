export TASK_NAME=mrpc

accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path facebook/bart-base \
    --dataset_name xsum \
    --num_train_epochs 25 \
    --output_dir ~/tmp/tst-summarization/xsum \
    --per_device_train_batch_size 16