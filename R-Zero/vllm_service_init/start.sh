model_path=$1
run_id=$2
export VLLM_DISABLE_COMPILE_CACHE=1
# STOCK 8-GPU: questioner verl trains on GPU 0-3; 4 solver-scoring servers run
# concurrently on GPU 4-7 (ports 5000-5003). caller_penalty.py shards the reward
# batch across these 4 servers (N_SERVERS=4).
CUDA_VISIBLE_DEVICES=4 python vllm_service_init/start_vllm_server.py --port 5000 --model_path $model_path &
CUDA_VISIBLE_DEVICES=5 python vllm_service_init/start_vllm_server.py --port 5001 --model_path $model_path &
CUDA_VISIBLE_DEVICES=6 python vllm_service_init/start_vllm_server.py --port 5002 --model_path $model_path &
CUDA_VISIBLE_DEVICES=7 python vllm_service_init/start_vllm_server.py --port 5003 --model_path $model_path &
