if [ "$1" == 'mt5' ]; then 
    echo "Training MT5"
    python train.py \
        --file_path ./data/train.jsonl \
        --output_dir ./checkpoints_mt5/ \
        --train_batch_size 16 \
        --lr 5e-4 \
        --gradient_accumulation_steps 4 \
        --eval_batch_size 32 \
        --n_epochs 40 
elif [ "$1" == 'gpt2' ]; then
    echo "Training GPT-2"
    python train_gpt.py \
        --file_path ./data/train.jsonl \
        --output_dir ./checkpoints_gpt_revised/ \
        --max_input_length 1024 \
        --max_out_length 64 \
        --train_batch_size 4 \
        --lr 1e-3 \
        --gradient_accumulation_steps 16 \
        --eval_batch_size 16 \
        --n_epochs 40
else
    echo "Invalid model name"
fi
