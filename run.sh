# if [ "$1" == "mt5" ]; then
python inference.py \
    --model_path ./checkpoints_mt5/ \
    --test_file_dir ${1} \
    --output_file_name ${2} \
    --max_input_length 256 \
    --max_out_length 64 \
    --test_batch_size 64 \
    --num_beams 10 \
    --type beam \
    --temp 0.6 \
    --top_k 30 \
    --top_p 0.8
# elif [ "$1" == "gpt2" ]; then
#     python inference_gpt.py \
#         --model_path ./checkpoints_gpt \
#         --test_file_dir ./data/public.jsonl \
#         --max_input_length 1024 \
#         --max_out_length 64 \
#         --test_batch_size 8 \
#         --type beam \
#         --num_beams 10 \
#         --temp 0.6 \
#         --top_k 10 \
#         --top_p 0.95
# else
#     echo "Usage: bash run.sh [mt5|gpt2]"
# fi
