# load the model name from the command line
model_name=$1
num_samples=$2
save_name=$3
export VLLM_DISABLE_COMPILE_CACHE=1
# GPU-0-3: 4 shards (suffix 0-3) on GPU 0-3. upload.py consumes exactly suffix 0-3,
# so 4 shards x num_samples reproduces the effective 4000-question pool the stock
# 8-GPU run actually used (shards 4-7 were generated but discarded by upload).
CUDA_VISIBLE_DEVICES=0 python question_generate/question_generate.py --model $model_name --suffix 0 --num_samples $num_samples --save_name $save_name &
CUDA_VISIBLE_DEVICES=1 python question_generate/question_generate.py --model $model_name --suffix 1 --num_samples $num_samples --save_name $save_name &
CUDA_VISIBLE_DEVICES=2 python question_generate/question_generate.py --model $model_name --suffix 2 --num_samples $num_samples --save_name $save_name &
CUDA_VISIBLE_DEVICES=3 python question_generate/question_generate.py --model $model_name --suffix 3 --num_samples $num_samples --save_name $save_name &

wait
