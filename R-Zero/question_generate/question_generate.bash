# load the model name from the command line
model_name=$1
num_samples=$2
save_name=$3
export VLLM_DISABLE_COMPILE_CACHE=1
# STOCK 8-GPU: 8 data-parallel shards (suffix 0-7) on GPU 0-7. upload.py consumes suffix 0-7.
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g python question_generate/question_generate.py --model $model_name --suffix $g --num_samples $num_samples --save_name $save_name &
done
wait
