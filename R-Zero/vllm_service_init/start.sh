model_path=$1
run_id=$2
export VLLM_DISABLE_COMPILE_CACHE=1
# GPU-0-3 (2+2) layout: questioner verl trains on GPU 0,1; these 2 solver-scoring
# servers run concurrently on GPU 2,3 (ports 5000,5001). caller_penalty.py shards
# the reward batch across exactly these 2 servers (range(2)).
CUDA_VISIBLE_DEVICES=2 python vllm_service_init/start_vllm_server.py --port 5000 --model_path $model_path &
CUDA_VISIBLE_DEVICES=3 python vllm_service_init/start_vllm_server.py --port 5001 --model_path $model_path &