model_path=$1
run_id=$2
export VLLM_DISABLE_COMPILE_CACHE=1
# 8-GPU: questioner verl on GPU 0-3; 4 solver-scoring servers on GPU 4-7 (ports 5000-5003).
CUDA_VISIBLE_DEVICES=4 python vllm_service_init/start_vllm_server.py --port 5000 --model_path $model_path &
CUDA_VISIBLE_DEVICES=5 python vllm_service_init/start_vllm_server.py --port 5001 --model_path $model_path &
CUDA_VISIBLE_DEVICES=6 python vllm_service_init/start_vllm_server.py --port 5002 --model_path $model_path &
CUDA_VISIBLE_DEVICES=7 python vllm_service_init/start_vllm_server.py --port 5003 --model_path $model_path &
