#!/bin/bash

solver_model_path=$1
questioner_model_path=$2
save_path=$3
echo "save_path: $save_path"
# step knobs (default = proven 6 steps save_freq 1 -> global_step_5;
# smoke overrides: RZ_Q_MAXSTEPS=2 RZ_Q_SAVEFREQ=1 RZ_Q_GSTEP=1 -> global_step_1)
Q_MAXSTEPS=${RZ_Q_MAXSTEPS:-6}
Q_SAVEFREQ=${RZ_Q_SAVEFREQ:-1}
Q_GSTEP=${RZ_Q_GSTEP:-5}
# 生成唯一 RUN_ID
RUN_ID=$(date +%s%N)
export RUN_ID

echo "RUN_ID=$RUN_ID"

# 启动 vllm 服务（记录 PID）
bash vllm_service_init/start.sh $solver_model_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"

# 开始训练 Questioner
echo "Start training questioner: $questioner_model_path -> $save_path"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$questioner_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./examples/reward_function/caller_penalty.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=4 \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    worker.rollout.n=4 \
    data.rollout_batch_size=${RZ_Q_ROLLOUT:-512} \
    worker.actor.global_batch_size=16 \
    trainer.max_steps=${Q_MAXSTEPS} \
    trainer.save_freq=${Q_SAVEFREQ}

sleep 5

# 合并模型
echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_${Q_GSTEP}/actor

sleep 10

pkill python

echo "questioner training finished"
