solver_model_path=$1
questioner_model_path=$2
experiment_name=$3

# step knobs (default = proven 20 steps -> checkpoint at global_step_15;
# smoke overrides: RZ_S_MAXSTEPS=2 RZ_S_SAVEFREQ=1 RZ_S_GSTEP=1 -> global_step_1)
S_MAXSTEPS=${RZ_S_MAXSTEPS:-20}
S_GSTEP=${RZ_S_GSTEP:-15}
SF=""; [ -n "${RZ_S_SAVEFREQ:-}" ] && SF="trainer.save_freq=${RZ_S_SAVEFREQ}"

echo $STORAGE_PATH
echo "start train solver $experiment_name $solver_model_path $questioner_model_path"

export VLLM_DISABLE_COMPILE_CACHE=1
echo 'start generate question'
bash question_generate/question_generate.bash $questioner_model_path ${RZ_NUM_SAMPLES:-1000} $experiment_name
echo 'start evaluate generated question'
bash question_evaluate/evaluate.sh $solver_model_path $experiment_name
echo 'start upload'
python question_evaluate/upload.py --repo_name ${experiment_name} --max_score 0.8 --min_score 0.3 --experiment_name ${experiment_name}
echo 'start train'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    data.train_files=${HUGGINGFACENAME}/${experiment_name}@train \
    trainer.total_epochs=100 \
    trainer.max_steps=${S_MAXSTEPS} \
    trainer.n_gpus_per_node=8 \
    data.rollout_batch_size=${RZ_S_ROLLOUT:-64} \
    worker.actor.global_batch_size=16 \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    ${SF}

echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_${S_GSTEP}/actor

sleep 10

echo "solver training finished"

bash evaluation/evaluate.bash ${STORAGE_PATH}/models/${experiment_name}/global_step_${S_GSTEP}/actor/huggingface
